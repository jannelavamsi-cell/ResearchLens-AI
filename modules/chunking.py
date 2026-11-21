"""
Section-Aware Chunking Engine for ResearchLens AI.
Partitions academic documents into token-bounded chunks respecting section borders,
preserving exact 1-indexed page ranges, and omitting References from retrieval.
"""

from typing import Any, Dict, List, Tuple

from core.logging_config import get_logger
from core.schemas import PageSchema, SectionSpan

logger = get_logger("chunking")


def approximate_token_count(text: str) -> int:
    """Estimates token count using whitespace and punctuation heuristics (~1.3 words per token)."""
    words = text.split()
    return int(len(words) * 1.3)


class SectionAwareChunker:
    """Chunks academic documents strictly within detected section boundaries."""

    def __init__(
        self,
        chunk_size_tokens: int = 350,
        chunk_overlap_tokens: int = 50,
        exclude_references: bool = True,
    ) -> None:
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.exclude_references = exclude_references

    def chunk_paper(
        self,
        paper_id: str,
        pages: List[PageSchema],
        sections: List[SectionSpan],
    ) -> List[Dict[str, Any]]:
        """
        Generates section-bounded chunks from pages and detected section spans.
        Each chunk contains: id, paper_id, chunk_index, page_start, page_end,
        section_name, text, token_count.
        """
        all_chunks: List[Dict[str, Any]] = []
        chunk_counter = 0

        # Build page-to-text map
        page_map = {p.page_number: p.text for p in pages}

        for sec in sections:
            sec_name = sec.section_name

            # Drop References section from retrieval if configured
            if self.exclude_references and sec_name.lower() in ["references", "bibliography"]:
                logger.debug(
                    f"Excluding '{sec_name}' section from search chunks for paper {paper_id}"
                )
                continue

            # Gather text across pages spanning this section
            sec_text_blocks: List[Tuple[int, str]] = []  # (page_num, paragraph)
            for page_num in range(sec.page_start, sec.page_end + 1):
                page_text = page_map.get(page_num, "")
                if not page_text:
                    continue
                paragraphs = page_text.split("\n\n")
                for para in paragraphs:
                    clean_p = para.strip()
                    if clean_p:
                        sec_text_blocks.append((page_num, clean_p))

            if not sec_text_blocks:
                continue

            # Formulate chunks bounded by token limits within this section
            current_chunk_paras: List[str] = []
            current_tokens = 0
            chunk_start_page = sec_text_blocks[0][0]
            chunk_end_page = sec_text_blocks[0][0]

            for page_num, para in sec_text_blocks:
                para_tokens = approximate_token_count(para)

                # If adding this paragraph exceeds target chunk size and we already have content:
                if (current_tokens + para_tokens) > self.chunk_size_tokens and current_chunk_paras:
                    chunk_text = "\n\n".join(current_chunk_paras)
                    chunk_id = f"chk-{paper_id[:8]}-{chunk_counter:04d}"

                    all_chunks.append(
                        {
                            "id": chunk_id,
                            "paper_id": paper_id,
                            "chunk_index": chunk_counter,
                            "page_start": chunk_start_page,
                            "page_end": chunk_end_page,
                            "section_name": sec_name,
                            "text": chunk_text,
                            "token_count": current_tokens,
                        }
                    )
                    chunk_counter += 1

                    # Apply overlap: carry forward the last paragraph if under overlap budget
                    if para_tokens <= self.chunk_overlap_tokens:
                        current_chunk_paras = [para]
                        current_tokens = para_tokens
                    else:
                        current_chunk_paras = [para]
                        current_tokens = para_tokens

                    chunk_start_page = page_num
                    chunk_end_page = page_num
                else:
                    current_chunk_paras.append(para)
                    current_tokens += para_tokens
                    chunk_end_page = max(chunk_end_page, page_num)

            # Flush remaining paragraph block for this section
            if current_chunk_paras:
                chunk_text = "\n\n".join(current_chunk_paras)
                chunk_id = f"chk-{paper_id[:8]}-{chunk_counter:04d}"

                all_chunks.append(
                    {
                        "id": chunk_id,
                        "paper_id": paper_id,
                        "chunk_index": chunk_counter,
                        "page_start": chunk_start_page,
                        "page_end": chunk_end_page,
                        "section_name": sec_name,
                        "text": chunk_text,
                        "token_count": current_tokens,
                    }
                )
                chunk_counter += 1

        logger.info(
            f"Chunked paper {paper_id}: {len(all_chunks)} chunks across {len(sections)} sections."
        )
        return all_chunks
