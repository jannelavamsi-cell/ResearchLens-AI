"""
PDF Processing Engine for ResearchLens AI.
Extracts page text with PyMuPDF block sorting for two-column academic layouts,
preserves 1-indexed page boundaries, performs header/footer removal,
detects scanned pages with graceful OCR fallback, and extracts tables.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf

from core.logging_config import get_logger
from core.schemas import PageSchema
from modules.results_tables import is_likely_results_table, normalize_table_dict
from modules.text_preprocessing import clean_page_text, detect_repeated_headers_footers

logger = get_logger("pdf_processor")


class PDFProcessor:
    """Processes academic PDF documents into structured pages and tables."""

    def __init__(self, enable_ocr: bool = True) -> None:
        self.enable_ocr = enable_ocr
        self._tesseract_available: Optional[bool] = None

    def _check_tesseract(self) -> bool:
        """Checks if pytesseract and the Tesseract binary are installed."""
        if self._tesseract_available is not None:
            return self._tesseract_available

        try:
            import pytesseract

            # Test invocation
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
        except Exception:
            self._tesseract_available = False
            logger.info(
                "Tesseract OCR is not installed. Scanned pages will be flagged without OCR."
            )

        return self._tesseract_available

    def _ocr_page(self, page: pymupdf.Page) -> str:
        """Attempts OCR on a scanned page pixmap if pytesseract is available."""
        if not self._check_tesseract():
            return ""

        try:
            import io

            import pytesseract
            from PIL import Image

            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img)
            return ocr_text.strip()
        except Exception as e:
            logger.warning(f"OCR failed on page {page.number + 1}: {e}")
            return ""

    def _extract_page_text_two_column(self, page: pymupdf.Page) -> str:
        """
        Extracts text from a page respecting two-column academic reading order.
        Uses PyMuPDF's built-in block sorting and column-aware heuristics.
        """
        # PyMuPDF sort=True sorts blocks in natural reading order
        blocks = page.get_text("blocks", sort=True)
        if not blocks:
            return ""

        page_width = page.rect.width
        midpoint = page_width / 2.0

        # Heuristic: Check if blocks span left and right columns distinctly
        has_two_columns = False
        left_blocks = []
        right_blocks = []
        spanning_blocks = []

        for b in blocks:
            # b: (x0, y0, x1, y1, text, block_no, block_type)
            if b[6] != 0:  # Skip image blocks
                continue
            text = b[4].strip()
            if not text:
                continue

            x0, y0, x1, _y1 = b[0], b[1], b[2], b[3]
            # Spanning banner (e.g. Title, Abstract)
            if x0 < (midpoint - 40) and x1 > (midpoint + 40):
                spanning_blocks.append((y0, text))
            elif x1 <= (midpoint + 20):
                left_blocks.append((y0, text))
            else:
                right_blocks.append((y0, text))

        if left_blocks and right_blocks:
            has_two_columns = True

        if has_two_columns:
            # Sort top banners, then entire left column top-to-bottom, then entire right column
            spanning_sorted = [t for _, t in sorted(spanning_blocks, key=lambda x: x[0])]
            left_sorted = [t for _, t in sorted(left_blocks, key=lambda x: x[0])]
            right_sorted = [t for _, t in sorted(right_blocks, key=lambda x: x[0])]
            full_blocks = spanning_sorted + left_sorted + right_sorted
        else:
            # Single column: sort by y0 coordinate
            full_blocks = [b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()]

        return "\n\n".join(full_blocks)

    def process_pdf(
        self, file_path: str
    ) -> Tuple[List[PageSchema], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Parses a PDF file into clean PageSchema objects, extracted tables, and metadata.
        Returns: (pages, tables, scan_report)
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        doc = pymupdf.open(str(path))
        raw_pages: List[Tuple[int, str, bool]] = []  # (page_num, text, is_ocr)
        extracted_tables: List[Dict[str, Any]] = []
        scanned_pages: List[int] = []

        # Pass 1: Extract raw block text per page
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1

            page_text = self._extract_page_text_two_column(page)
            is_ocr = False

            # Scanned page detection (< 100 characters of text)
            if len(page_text.strip()) < 100:
                scanned_pages.append(page_num)
                is_ocr = True
                if self.enable_ocr:
                    ocr_res = self._ocr_page(page)
                    if ocr_res:
                        page_text = ocr_res

            raw_pages.append((page_num, page_text, is_ocr))

            # Attempt Table Extraction via PyMuPDF find_tables()
            try:
                tables = page.find_tables()
                for table_idx, tab in enumerate(tables):
                    df_data = tab.extract()
                    if df_data and len(df_data) > 1:
                        raw_headers = [
                            str(c).strip() if c else f"Col_{i+1}"
                            for i, c in enumerate(df_data[0])
                        ]
                        cleaned_rows = [
                            [str(c).strip() if c is not None else "" for c in row]
                            for row in df_data[1:]
                        ]
                        is_res = is_likely_results_table(raw_headers, cleaned_rows)
                        extracted_tables.append(
                            normalize_table_dict(
                                {
                                    "page_number": page_num,
                                    "table_index": table_idx + 1,
                                    "headers": raw_headers,
                                    "header": raw_headers,
                                    "rows": cleaned_rows,
                                    "row_count": len(cleaned_rows),
                                    "col_count": len(raw_headers),
                                    "caption": "",
                                    "is_results_table": is_res,
                                }
                            )
                        )
            except Exception as e:
                logger.debug(f"Table extraction skipped on page {page_num}: {e}")

        # Pass 2: Detect recurring headers/footers across pages
        all_texts = [p[1] for p in raw_pages]
        repeated_headers = detect_repeated_headers_footers(all_texts, threshold=0.6)

        # Pass 3: Clean text and build final PageSchema objects
        cleaned_pages: List[PageSchema] = []
        for page_num, raw_text, is_ocr in raw_pages:
            cleaned = clean_page_text(raw_text, repeated_headers=repeated_headers)
            cleaned_pages.append(
                PageSchema(
                    page_number=page_num,
                    text=cleaned,
                    is_ocr=is_ocr,
                )
            )

        doc.close()

        scan_report = {
            "total_pages": len(cleaned_pages),
            "scanned_pages_count": len(scanned_pages),
            "scanned_pages": scanned_pages,
            "has_scanned_pages": len(scanned_pages) > 0,
            "ocr_applied": self._tesseract_available is True and len(scanned_pages) > 0,
        }

        logger.info(
            f"Processed PDF '{path.name}': {len(cleaned_pages)} pages, "
            f"{len(scanned_pages)} scanned pages, {len(extracted_tables)} tables."
        )

        return cleaned_pages, extracted_tables, scan_report
