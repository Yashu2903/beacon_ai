import argparse
from uuid import UUID

from app.core.database import SessionLocal
from app.services.llm_evidence_packet import build_page_evidence_packets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    document_id = UUID(args.document_id)

    db = SessionLocal()

    try:
        packets = build_page_evidence_packets(db=db, document_id=document_id)

        print(f"pages: {len(packets)}")

        total_text = 0
        total_diagrams = 0
        full_pages = 0
        warning_count = 0

        for packet in packets:
            total_text += len(packet.text_items)
            total_diagrams += len(packet.diagram_items)
            warning_count += len(packet.warnings)

            if packet.full_page_image and packet.full_page_image.local_path:
                full_pages += 1

            print(
                f"page={packet.page_number} "
                f"text={len(packet.text_items)} "
                f"diagrams={len(packet.diagram_items)} "
                f"full_page={'yes' if packet.full_page_image else 'no'} "
                f"warnings={len(packet.warnings)}"
            )

        print("---- summary ----")
        print(f"full_page_images: {full_pages}")
        print(f"text_items: {total_text}")
        print(f"diagram_items: {total_diagrams}")
        print(f"warnings: {warning_count}")

    finally:
        db.close()


if __name__ == "__main__":
    main()