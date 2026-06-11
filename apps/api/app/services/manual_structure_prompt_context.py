from uuid import UUID

from sqlalchemy.orm import Session

from app.models.manual_page_structure import ManualPageStructure


def build_manual_structure_prompt_context(
    db: Session,
    document_id: UUID,
) -> dict:
    """
    Builds compact manual-structure context for the step extraction LLM.

    This context tells the LLM:
    - which pages are assembly pages
    - which pages are inventory/info/back matter
    - which visible step numbers appear on each page

    The extractor should use this as a planning map before writing steps.
    """
    rows = (
        db.query(ManualPageStructure)
        .filter(ManualPageStructure.document_id == document_id)
        .order_by(ManualPageStructure.page_number.asc())
        .all()
    )

    pages = []

    assembly_pages = []
    non_assembly_pages = []
    visible_step_numbers_by_page = {}

    for row in rows:
        page_type = row.page_type.value if hasattr(row.page_type, "value") else str(row.page_type)

        visible_numbers = [
            int(number)
            for number in (row.visible_step_numbers or [])
            if str(number).isdigit()
        ]

        page_context = {
            "page_number": row.page_number,
            "page_type": page_type,
            "visible_step_numbers": visible_numbers,
            "confidence": row.confidence,
            "metadata": row.metadata_json or {},
        }

        pages.append(page_context)

        if visible_numbers:
            visible_step_numbers_by_page[str(row.page_number)] = visible_numbers

        if page_type in {"assembly_step", "mixed_inventory_and_step"}:
            assembly_pages.append(row.page_number)

        if page_type in {"cover", "parts_inventory", "informational", "back_matter"}:
            non_assembly_pages.append(row.page_number)

    all_visible_numbers = sorted(
        {
            number
            for numbers in visible_step_numbers_by_page.values()
            for number in numbers
        }
    )

    return {
        "has_manual_structure": bool(rows),
        "source": "ManualPageStructure",
        "pages": pages,
        "assembly_pages": assembly_pages,
        "non_assembly_pages": non_assembly_pages,
        "visible_step_numbers_by_page": visible_step_numbers_by_page,
        "all_visible_step_numbers": all_visible_numbers,
        "extraction_guidance": [
            "Use visible_step_numbers_by_page as the primary step-number map.",
            "Only extract numbered assembly steps from assembly_step or mixed_inventory_and_step pages.",
            "Do not create numbered assembly steps from cover, parts_inventory, informational, or back_matter pages.",
            "If a page has visible_step_numbers [8], produce one Step 8 unless the manual clearly shows separate numbered steps.",
            "If one visible step contains orientation guidance plus assembly action, merge them into one step.",
            "Do not invent missing step numbers just to fill sequence gaps.",
            "If a page is mixed_inventory_and_step, separate inventory listing from the actual visible assembly step.",
            "If the visual evidence is unclear, mark the step confidence lower and include needs_attention rather than inventing details.",
        ],
    }