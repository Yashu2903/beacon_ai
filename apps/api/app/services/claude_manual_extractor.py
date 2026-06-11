import base64
import json
import logging
import mimetypes
from pathlib import Path
from uuid import UUID

from anthropic import Anthropic
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.schemas.llm_manual import (
    LLMManualExtractionResult,
    LLMPageExtractionResponse,
)
from app.services.llm_evidence_packet import build_llm_evidence_packet


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are an expert assembly instruction parser for furniture manuals.

Your job is to extract assembly steps from manual evidence as structured JSON.

The manual may be highly visual or entirely visual, with little or no text. Interpret diagrams, arrows, icons, numbered circles, part illustrations, repeated-action symbols, check marks, and X marks carefully.

You may receive manual_structure_context in the evidence packet. This context is generated before extraction and describes the detected page type and visible step numbers for the manual.

Manual structure rules:

* If manual_structure_context.has_manual_structure is true, treat manual_structure_context.visible_step_numbers_by_page as the primary map of visible assembly step numbers.
* Use manual_structure_context.pages to understand the page_type for the current page.
* Only extract numbered assembly steps from pages classified as assembly_step or mixed_inventory_and_step.
* Do not create numbered assembly steps from pages classified as cover, parts_inventory, informational, or back_matter.
* If the current page is mixed_inventory_and_step, separate the inventory/parts list from the actual assembly step. Extract only the real visible assembly action.
* If the current page has visible_step_numbers, extracted step_number values should come from those visible numbers.
* If the current page has visible_step_numbers [8], produce one Step 8 unless the page clearly shows multiple independent visible numbered steps.
* If one visible step contains orientation guidance, warning panels, correct/incorrect examples, and the physical action, merge them into the same step. Do not split one visible step into duplicate step numbers.
* Do not invent missing step numbers to fill sequence gaps.
* If a page is informational, variant-only, safety-only, warranty-only, or back matter, set contains_assembly_steps to false even if it contains furniture diagrams.
* If a visible step number exists but the action is unclear, extract the visually supported action with lower confidence and explain the uncertainty in needs_attention_reason. Do not fabricate parts, tools, or actions.
* The number of extracted numbered steps should generally match the visible step numbers for the page unless visual evidence clearly says otherwise.

Visual interpretation rules:

* Arrows usually indicate motion, insertion, rotation, alignment, attachment, tightening, or direction.
* X marks usually indicate warnings, incorrect actions, or prohibited orientation.
* Check marks usually indicate correct orientation or expected result.
* Numbered circles may indicate either step numbers or part identifiers. Use surrounding evidence and manual_structure_context to decide.
* Repeated symbols like "x2", "x4", or repeated part drawings indicate quantity.
* Tool icons indicate required tools.
* If the page shows only inventory, warranty, cover content, or safety information with no assembly action, set contains_assembly_steps to false.

Extract only information supported by the provided evidence. Do not invent parts, tools, quantities, warnings, orientations, or expected results.

Return only valid JSON. Do not include markdown. Do not include explanation outside the JSON.

For this page, output:

{
  "page_number": <integer>,
  "contains_assembly_steps": <true or false>,
  "extracted_steps": [
    {
      "step_number": <integer or null>,
      "task_title": <short task title or null>,
      "action": <clear description of the assembly action>,
      "parts": [<part names or IDs visible in this step>],
      "tools": [<tools required, empty array if none>],
      "quantities": [<quantity descriptions such as "4 screws" or "repeat 2 times">],
      "orientation": <orientation detail if visible, otherwise null>,
      "warning": <warning or incorrect-action note if shown, otherwise null>,
      "expected_result": <what the assembly should look like after this step>,
      "source_evidence_ids": [<all evidence IDs used for this step>],
      "visual_evidence_ids": [<diagram_region evidence IDs used for this step>],
      "text_evidence_ids": [<page_text_span evidence IDs used for this step>],
      "confidence": <0.0 to 1.0>,
      "needs_attention_reason": <why this step is uncertain, otherwise null>
    }
  ],
  "page_summary": <short summary of what this page contains>,
  "unresolved_questions": [<anything ambiguous or unclear>],
  "confidence": <0.0 to 1.0>
}

Rules:

* Use only the evidence IDs provided in the page evidence JSON.
* Every extracted step must include at least one source_evidence_id.
* visual_evidence_ids must come only from diagram evidence.
* text_evidence_ids must come only from text evidence.
* If an action is unclear, still describe what is visually shown, but set confidence below 0.5 and explain the uncertainty in needs_attention_reason.
* If two diagrams on the same page show one continuous action, treat them as one step.
* If multiple independent visible numbered assembly actions appear on one page, extract them as separate steps.
* Preserve the manual's visible order on the page.
* Do not restart or force global step numbering. Use visible step numbers when available; otherwise set step_number to null. The backend will normalize global numbering after all pages are processed.
* Do not duplicate a visible step number on the same page unless the manual clearly repeats the same step number for separate alternative variants.
* Do not extract warning-only panels as separate assembly steps. Attach them to the relevant step's warning or orientation field.
* Do not extract parts inventory rows as assembly steps.
* Do not extract back-matter product views, care instructions, safety pages, or variant pages as assembly steps.
"""


def _page_prompt(page_packet: dict, manual_structure_context: dict) -> str:
    response_schema = {
        "page_number": page_packet["page_number"],
        "contains_assembly_steps": True,
        "extracted_steps": [
            {
                "step_number": 1,
                "task_title": "Furniture Assembly",
                "action": "Attach part A to part B.",
                "parts": ["part A", "part B"],
                "tools": ["Allen key"],
                "quantities": ["4 screws"],
                "orientation": "holes facing inward",
                "warning": None,
                "expected_result": "Part A is secured to part B.",
                "source_evidence_ids": ["uuid-1", "uuid-2"],
                "visual_evidence_ids": ["diagram-region-uuid"],
                "text_evidence_ids": ["text-span-uuid"],
                "confidence": 0.85,
                "needs_attention_reason": None,
            }
        ],
        "page_summary": "Short page summary.",
        "unresolved_questions": [],
        "confidence": 0.85,
    }

    evidence_json = json.dumps(
        {
            "manual_structure_context": manual_structure_context,
            "page": page_packet,
        },
        indent=2,
    )

    return (
        "Extract structured furniture assembly steps from this page evidence.\n\n"
        "Return JSON exactly matching this shape:\n"
        f"{json.dumps(response_schema, indent=2)}\n\n"
        "Important evidence rules:\n"
        "- Every extracted step must include at least one source_evidence_id.\n"
        "- visual_evidence_ids must come only from diagram_evidence IDs.\n"
        "- text_evidence_ids must come only from text_evidence IDs.\n"
        "- If the page has no actual assembly step, return contains_assembly_steps=false and extracted_steps=[].\n"
        "- If the page is ambiguous, keep confidence low and explain in unresolved_questions.\n\n"
        "Evidence packet JSON:\n"
        f"{evidence_json}"
    )


def _guess_media_type(path: str) -> str:
    media_type, _ = mimetypes.guess_type(path)

    if media_type in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        return media_type

    suffix = Path(path).suffix.lower()

    if suffix == ".png":
        return "image/png"

    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"

    return "image/png"


def _image_block(path: str) -> dict | None:
    image_path = Path(path)

    if not image_path.exists():
        return None

    try:
        image_bytes = image_path.read_bytes()
    except Exception:
        return None

    encoded = base64.b64encode(image_bytes).decode("utf-8")

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": _guess_media_type(str(image_path)),
            "data": encoded,
        },
    }


def _storage_key_to_local_path(storage_key: str | None) -> str | None:
    if not storage_key:
        return None

    return str(Path(settings.local_storage_dir) / storage_key)


def _extract_json_text(response) -> str:
    parts: list[str] = []

    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)

    return "\n".join(parts).strip()


class ClaudeManualExtractor:
    def __init__(self) -> None:
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured")

        self.client = Anthropic(api_key=settings.anthropic_api_key)

    def extract_document(
        self,
        db: Session,
        document_id: UUID,
        job_id: UUID,
    ) -> LLMManualExtractionResult:
        evidence_packet = build_llm_evidence_packet(db=db, document_id=document_id)
        manual_structure_context = evidence_packet["manual_structure_context"]
        page_packets = evidence_packet["pages"]

        logger.info(
            "LLM extraction packet built document_id=%s has_manual_structure=%s visible_step_numbers_by_page=%s",
            document_id,
            manual_structure_context.get("has_manual_structure"),
            manual_structure_context.get("visible_step_numbers_by_page"),
        )

        pages: list[LLMPageExtractionResponse] = []
        warnings: list[str] = []

        fallback_used = False
        total_input_tokens = 0
        total_output_tokens = 0
        model_used = settings.claude_primary_model

        for page_packet in page_packets:
            try:
                page_response, usage = self.extract_page(
                    page_packet=page_packet,
                    manual_structure_context=manual_structure_context,
                    model_name=settings.claude_primary_model,
                )

                if usage:
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)

                if self.should_use_fallback(page_response, page_packet):
                    if settings.claude_enable_fallback:
                        fallback_used = True
                        model_used = settings.claude_fallback_model

                        fallback_response, fallback_usage = self.extract_page(
                            page_packet=page_packet,
                            manual_structure_context=manual_structure_context,
                            model_name=settings.claude_fallback_model,
                        )

                        if fallback_usage:
                            total_input_tokens += fallback_usage.get(
                                "input_tokens", 0
                            )
                            total_output_tokens += fallback_usage.get(
                                "output_tokens", 0
                            )

                        page_response = fallback_response
                    else:
                        warnings.append(
                            f"Page {page_packet['page_number']}: fallback suggested but disabled"
                        )

                pages.append(page_response)

            except Exception as exc:
                warnings.append(
                    f"Page {page_packet['page_number']}: primary extraction failed: {exc}"
                )

                if settings.claude_enable_fallback:
                    try:
                        fallback_used = True
                        model_used = settings.claude_fallback_model

                        fallback_response, fallback_usage = self.extract_page(
                            page_packet=page_packet,
                            manual_structure_context=manual_structure_context,
                            model_name=settings.claude_fallback_model,
                        )

                        if fallback_usage:
                            total_input_tokens += fallback_usage.get(
                                "input_tokens", 0
                            )
                            total_output_tokens += fallback_usage.get(
                                "output_tokens", 0
                            )

                        pages.append(fallback_response)

                    except Exception as fallback_exc:
                        warnings.append(
                            f"Page {page_packet['page_number']}: fallback extraction failed: {fallback_exc}"
                        )
                        pages.append(
                            LLMPageExtractionResponse(
                                page_number=page_packet["page_number"],
                                contains_assembly_steps=False,
                                extracted_steps=[],
                                page_summary=None,
                                unresolved_questions=[
                                    "Both primary and fallback LLM extraction failed."
                                ],
                                confidence=0.0,
                            )
                        )
                else:
                    pages.append(
                        LLMPageExtractionResponse(
                            page_number=page_packet["page_number"],
                            contains_assembly_steps=False,
                            extracted_steps=[],
                            page_summary=None,
                            unresolved_questions=[
                                "Primary LLM extraction failed and fallback is disabled."
                            ],
                            confidence=0.0,
                        )
                    )

        total_steps = sum(len(page.extracted_steps) for page in pages)

        return LLMManualExtractionResult(
            pages=pages,
            total_steps=total_steps,
            model_used=model_used,
            fallback_used=fallback_used,
            warnings=warnings,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

    def extract_page(
        self,
        page_packet: dict,
        manual_structure_context: dict,
        model_name: str,
    ) -> tuple[LLMPageExtractionResponse, dict]:
        content_blocks: list[dict] = []

        if settings.include_full_page_image_for_llm:
            full_page_image_path = _storage_key_to_local_path(
                page_packet.get("full_page_image_storage_key")
            )
            block = _image_block(full_page_image_path) if full_page_image_path else None
            if block:
                content_blocks.append(block)

        for item in page_packet.get("diagram_evidence", []):
            local_path = _storage_key_to_local_path(item.get("storage_key"))
            if local_path:
                block = _image_block(local_path)
                if block:
                    content_blocks.append(block)

        content_blocks.append(
            {
                "type": "text",
                "text": _page_prompt(
                    page_packet=page_packet,
                    manual_structure_context=manual_structure_context,
                ),
            }
        )

        response = self.client.messages.create(
            model=model_name,
            max_tokens=settings.claude_max_tokens,
            temperature=settings.claude_temperature,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": content_blocks,
                }
            ],
        )

        raw_text = _extract_json_text(response)

        try:
            parsed_json = json.loads(raw_text)
        except json.JSONDecodeError:
            # Sometimes models accidentally wrap JSON in stray text.
            # Try to recover the first JSON object.
            start = raw_text.find("{")
            end = raw_text.rfind("}")

            if start == -1 or end == -1 or end <= start:
                raise

            parsed_json = json.loads(raw_text[start : end + 1])

        try:
            parsed_response = LLMPageExtractionResponse.model_validate(parsed_json)
        except ValidationError as exc:
            raise ValueError(f"Claude response failed schema validation: {exc}") from exc

        usage = {}

        if getattr(response, "usage", None):
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(response.usage, "output_tokens", 0) or 0,
            }

        return parsed_response, usage

    def should_use_fallback(
        self,
        primary_response: LLMPageExtractionResponse,
        page_packet: dict,
    ) -> bool:
        serious_warning_types = {
            "missing_step_region",
            "missing_primary_step_region",
            "giant_region",
            "ocr_unavailable",
        }

        has_serious_warning = any(
            warning.get("warning_type") in serious_warning_types
            for warning in page_packet.get("warnings", [])
        )

        if has_serious_warning and primary_response.confidence < 0.80:
            return True

        if primary_response.contains_assembly_steps and not primary_response.extracted_steps:
            return True

        if primary_response.confidence < settings.llm_min_step_confidence:
            return True

        for step in primary_response.extracted_steps:
            if step.confidence < settings.llm_min_step_confidence:
                return True

            if not step.source_evidence_ids:
                return True

        return False
