import base64
import json
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
from app.services.llm_evidence_packet import (
    PageEvidencePacket,
    build_page_evidence_packets,
)


SYSTEM_PROMPT = """
You are an expert assembly instruction parser for furniture manuals.

Your job is to extract assembly steps from manual evidence as structured JSON.

The manual may be highly visual or entirely visual, with little or no text. Interpret diagrams, arrows, icons, numbered circles, part illustrations, repeated-action symbols, check marks, and X marks carefully.

Visual interpretation rules:

* Arrows usually indicate motion, insertion, rotation, alignment, attachment, tightening, or direction.
* X marks usually indicate warnings, incorrect actions, or prohibited orientation.
* Check marks usually indicate correct orientation or expected result.
* Numbered circles may indicate either step numbers or part identifiers. Use surrounding evidence to decide.
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
* If multiple independent actions appear on one page, extract them as separate steps.
* Preserve the manual's visible order on the page.
* Do not restart or force global step numbering. Use visible step numbers when available; otherwise set step_number to null. The backend will normalize global numbering after all pages are processed.

"""


def _page_prompt(packet: PageEvidencePacket) -> str:
    response_schema = {
        "page_number": packet.page_number,
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
        "Page evidence JSON:\n"
        f"{json.dumps(packet.to_prompt_json(), indent=2)}"
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
        packets = build_page_evidence_packets(db=db, document_id=document_id)

        pages: list[LLMPageExtractionResponse] = []
        warnings: list[str] = []

        fallback_used = False
        total_input_tokens = 0
        total_output_tokens = 0
        model_used = settings.claude_primary_model

        for packet in packets:
            try:
                page_response, usage = self.extract_page(
                    packet=packet,
                    model_name=settings.claude_primary_model,
                )

                if usage:
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)

                if self.should_use_fallback(page_response, packet):
                    if settings.claude_enable_fallback:
                        fallback_used = True
                        model_used = settings.claude_fallback_model

                        fallback_response, fallback_usage = self.extract_page(
                            packet=packet,
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
                            f"Page {packet.page_number}: fallback suggested but disabled"
                        )

                pages.append(page_response)

            except Exception as exc:
                warnings.append(
                    f"Page {packet.page_number}: primary extraction failed: {exc}"
                )

                if settings.claude_enable_fallback:
                    try:
                        fallback_used = True
                        model_used = settings.claude_fallback_model

                        fallback_response, fallback_usage = self.extract_page(
                            packet=packet,
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
                            f"Page {packet.page_number}: fallback extraction failed: {fallback_exc}"
                        )
                        pages.append(
                            LLMPageExtractionResponse(
                                page_number=packet.page_number,
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
                            page_number=packet.page_number,
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
        packet: PageEvidencePacket,
        model_name: str,
    ) -> tuple[LLMPageExtractionResponse, dict]:
        content_blocks: list[dict] = []

        if (
            settings.include_full_page_image_for_llm
            and packet.full_page_image
            and packet.full_page_image.local_path
        ):
            block = _image_block(packet.full_page_image.local_path)
            if block:
                content_blocks.append(block)

        for item in packet.diagram_items:
            if item.local_path:
                block = _image_block(item.local_path)
                if block:
                    content_blocks.append(block)

        content_blocks.append(
            {
                "type": "text",
                "text": _page_prompt(packet),
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
        packet: PageEvidencePacket,
    ) -> bool:
        serious_warning_types = {
            "missing_step_region",
            "missing_primary_step_region",
            "giant_region",
            "ocr_unavailable",
        }

        has_serious_warning = any(
            warning.get("warning_type") in serious_warning_types
            for warning in packet.warnings
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
