import json
from typing import Any
from uuid import UUID
from app.services.page_image_evidence import get_page_image_evidence_for_pages
import anthropic
from sqlalchemy.orm import Session

from app.core.config import settings
from app.schemas.step_repair import StepRepairProposal
from app.services.step_repair import build_or_apply_step_repair


STEP_REPAIR_SYSTEM_PROMPT = """
You are repairing extracted furniture assembly steps from an instruction manual.

You receive:
1. A validation summary.
2. A repair plan.
3. The current extracted steps.
4. Rules for safe repair.

Your job:
Return a JSON object only. Do not include markdown.

Allowed operations:
- merge_steps
- delete_step
- create_step
- edit_step
- renumber_step
- no_change
- needs_manual_review

Rules:
- Do not invent steps.
- Do not create sequence numbers only to fill gaps.
- Only create a missing step if the repair packet contains enough grounded evidence to write the action.
- If the exact action is not available, use needs_manual_review instead of create_step.
- If a step comes from an informational, variant-only, cover, inventory, or back-matter page, usually propose delete_step.
- If duplicate steps describe the same visible step, propose merge_steps.
- Preserve existing step IDs when editing existing steps.
- If unsure, use needs_manual_review or no_change and add a warning.
- Every proposed change must include a clear reason.
- The output must be valid JSON matching the expected schema.
"""


UNSAFE_CREATE_STEP_PHRASES = {
    "likely",
    "probably",
    "may involve",
    "might involve",
    "appears to involve",
    "requires human review",
    "human review required",
    "please inspect",
    "manual inspection",
    "cannot be confirmed",
    "could not be confirmed",
    "exact action not available",
    "exact instruction text not available",
    "requires manual inspection",
    "needs review",
    "needs manual review",
}


def _get_anthropic_client() -> anthropic.Anthropic:
    api_key = settings.anthropic_api_key

    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    return anthropic.Anthropic(api_key=api_key)


def _get_repair_model_name() -> str:
    return settings.claude_repair_model or settings.claude_primary_model


def _safe_json_loads(text: str) -> dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()

    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```").strip()

    return json.loads(cleaned)


def _build_user_prompt(repair_packet: dict) -> str:
    expected_schema = {
        "job_id": "string",
        "document_id": "string",
        "summary": "string",
        "can_apply_automatically": False,
        "requires_human_review": True,
        "proposed_changes": [
            {
                "operation": (
                    "merge_steps | delete_step | create_step | edit_step | "
                    "renumber_step | no_change | needs_manual_review"
                ),
                "reason": "string",
                "step_id": "string or null",
                "step_ids": ["string"],
                "step_number": "integer or null",
                "source_page_number": "integer or null",
                "order_index": "integer or null",
                "action": "string or null",
                "parts": ["string"],
                "tools": ["string"],
                "confidence": "number or null",
                "evidence_notes": "string or null",
                "metadata": {},
            }
        ],
        "warnings": ["string"],
    }

    return (
        "Repair packet:\n"
        f"{json.dumps(repair_packet, indent=2, default=str)}\n\n"
        "Return JSON only using this schema:\n"
        f"{json.dumps(expected_schema, indent=2)}"
    )


def _contains_unsafe_create_language(value: str | None) -> bool:
    if not value:
        return False

    lowered = value.lower()

    return any(phrase in lowered for phrase in UNSAFE_CREATE_STEP_PHRASES)


def _is_unsafe_create_step(change: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    A create_step is unsafe if Claude is clearly guessing.

    We only allow create_step when the action is specific, grounded, and confident.
    If the model says "likely", "requires human review", or provides a placeholder,
    we convert it into needs_manual_review.
    """
    reasons: list[str] = []

    if change.get("operation") != "create_step":
        return False, reasons

    action = change.get("action")
    reason = change.get("reason")
    evidence_notes = change.get("evidence_notes")
    confidence = change.get("confidence")

    parts = change.get("parts") or []
    tools = change.get("tools") or []

    if confidence is None:
        reasons.append("create_step has no confidence value")
    else:
        try:
            if float(confidence) < 0.70:
                reasons.append(f"create_step confidence is below threshold: {confidence}")
        except (TypeError, ValueError):
            reasons.append("create_step confidence is not numeric")

    if not action or not str(action).strip():
        reasons.append("create_step has empty action")

    if _contains_unsafe_create_language(action):
        reasons.append("create_step action contains uncertain/manual-review language")

    if _contains_unsafe_create_language(reason):
        reasons.append("create_step reason contains uncertain/manual-review language")

    if _contains_unsafe_create_language(evidence_notes):
        reasons.append("create_step evidence_notes contains uncertain/manual-review language")

    if not parts and not tools:
        reasons.append("create_step has empty parts and tools")

    metadata = change.get("metadata") or {}

    if metadata.get("requires_manual_inspection") is True:
        reasons.append("create_step metadata requires manual inspection")

    return len(reasons) > 0, reasons


def _convert_unsafe_create_to_manual_review(
    change: dict[str, Any],
    safety_reasons: list[str],
) -> dict[str, Any]:
    step_number = change.get("step_number")
    source_page_number = change.get("source_page_number")

    existing_reason = change.get("reason") or ""

    return {
        "operation": "needs_manual_review",
        "reason": (
            f"Visible Step {step_number} may exist on page {source_page_number}, "
            "but the proposed create_step was not safe to apply automatically. "
            "The exact action must be confirmed from page-level manual evidence before creating this step. "
            f"Original model reason: {existing_reason}"
        ),
        "step_id": None,
        "step_ids": change.get("step_ids") or [],
        "step_number": step_number,
        "source_page_number": source_page_number,
        "order_index": change.get("order_index"),
        "action": None,
        "parts": [],
        "tools": [],
        "confidence": change.get("confidence"),
        "evidence_notes": (
            "Converted from create_step to needs_manual_review by safety guard. "
            f"Safety reasons: {'; '.join(safety_reasons)}"
        ),
        "metadata": {
            **(change.get("metadata") or {}),
            "safety_guard_converted": True,
            "original_operation": "create_step",
            "safety_reasons": safety_reasons,
            "original_action": change.get("action"),
            "original_parts": change.get("parts") or [],
            "original_tools": change.get("tools") or [],
        },
    }


def _apply_repair_proposal_safety_guard(
    parsed: dict[str, Any],
) -> dict[str, Any]:
    """
    Post-process Claude output before Pydantic validation.

    This prevents unsafe guessed create_step proposals from becoming real repair patches.
    """
    proposed_changes = parsed.get("proposed_changes") or []
    safe_changes: list[dict[str, Any]] = []
    safety_warnings: list[str] = []

    for change in proposed_changes:
        is_unsafe, safety_reasons = _is_unsafe_create_step(change)

        if is_unsafe:
            safe_change = _convert_unsafe_create_to_manual_review(
                change=change,
                safety_reasons=safety_reasons,
            )

            safe_changes.append(safe_change)

            safety_warnings.append(
                f"Converted unsafe create_step for Step {change.get('step_number')} "
                f"on page {change.get('source_page_number')} to needs_manual_review: "
                f"{'; '.join(safety_reasons)}"
            )
        else:
            safe_changes.append(change)

    parsed["proposed_changes"] = safe_changes

    existing_warnings = parsed.get("warnings") or []
    parsed["warnings"] = existing_warnings + safety_warnings

    if safety_warnings:
        parsed["can_apply_automatically"] = False
        parsed["requires_human_review"] = True

        summary = parsed.get("summary") or ""
        parsed["summary"] = (
            summary
            + " Safety guard converted one or more unsafe create_step proposals "
            "into needs_manual_review."
        ).strip()

    return parsed


def propose_step_repair_with_claude(
    db: Session,
    job_id: UUID,
) -> StepRepairProposal:
    repair_response = build_or_apply_step_repair(
        db=db,
        job_id=job_id,
        apply_safe_fixes=False,
    )

    repair_packet = repair_response["llm_repair_packet"]

    client = _get_anthropic_client()
    model = _get_repair_model_name()

    candidate_pages = _get_candidate_pages_from_repair_packet(repair_packet)

    page_images = get_page_image_evidence_for_pages(
        db=db,
        document_id=UUID(repair_response["document_id"]),
        page_numbers=candidate_pages,
        max_images_per_page=1,
    )

    user_content = _build_multimodal_user_content(
        repair_packet=repair_packet,
        page_images=page_images,
    )

    message = client.messages.create(
        model=model,
        max_tokens=settings.claude_repair_max_tokens,
        temperature=settings.claude_repair_temperature,
        system=STEP_REPAIR_SYSTEM_PROMPT,
        messages=[
          {
            "role": "user",
            "content": user_content,
          }
        ],
    )

    response_text = ""

    for block in message.content:
        if getattr(block, "type", None) == "text":
            response_text += block.text

    parsed = _safe_json_loads(response_text)

    parsed["job_id"] = str(job_id)
    parsed["document_id"] = repair_response["document_id"]

    parsed = _apply_repair_proposal_safety_guard(parsed)

    parsed["raw_model_response"] = {
        "model": model,
        "stop_reason": message.stop_reason,
        "usage": {
            "input_tokens": getattr(message.usage, "input_tokens", None),
            "output_tokens": getattr(message.usage, "output_tokens", None),
        },
        "page_image_count": len(page_images),
        "candidate_pages": candidate_pages,
    }

    return StepRepairProposal.model_validate(parsed)

def _get_candidate_pages_from_repair_packet(repair_packet: dict) -> list[int]:
    pages: set[int] = set()

    for item in repair_packet.get("repair_plan", []):
        if item.get("repair_type") not in {
            "create_missing_visible_step",
            "remove_step_from_non_assembly_page",
        }:
            continue

        page_number = item.get("page_number")

        if isinstance(page_number, int):
            pages.add(page_number)

        for candidate_page in item.get("candidate_pages", []):
            if isinstance(candidate_page, int):
                pages.add(candidate_page)

    return sorted(pages)


def _build_multimodal_user_content(
    repair_packet: dict,
    page_images: list[dict],
) -> list[dict]:
    text_prompt = _build_user_prompt(repair_packet)

    if page_images:
        text_prompt += (
            "\n\nPage images are attached below. Use them only as visual evidence. "
            "If the page image does not clearly support a create_step action, use needs_manual_review."
        )

    content: list[dict] = [
        {
            "type": "text",
            "text": text_prompt,
        }
    ]

    for image in page_images:
        content.append(
            {
                "type": "text",
                "text": (
                    f"Attached image evidence: document page {image['page_number']} "
                    f"from evidence {image['evidence_id']}."
                ),
            }
        )

        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image["media_type"],
                    "data": image["base64"],
                },
            }
        )

    return content