"""
문서 파서
- document_loader 결과를 구조화
- 메타데이터 유지 (page, section, article 등)
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ─────────────────────────────────────────────────────────────
# 데이터 구조
# ─────────────────────────────────────────────────────────────

@dataclass
class ContentBlock:
    """문서의 의미 단위 블록"""
    text: str
    block_type: str                  # title, page, paragraph, article 등
    level: int = 0
    page: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """파싱된 문서"""
    text: str
    blocks: List[ContentBlock]
    metadata: Dict                   # file_name, file_type, title 등


# ─────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────

def extract_title(text: str) -> str:
    """문서 제목 추출"""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:5]:
        if line.lower().startswith(("title:", "제목:")):
            return line.split(":", 1)[1].strip()
        if len(line) > 5:
            return line[:100]
    return "제목 없음"


# ─────────────────────────────────────────────────────────────
# 기본 파서 (plain text)
# ─────────────────────────────────────────────────────────────

def parse_plain_text(
    text: str,
    file_name: str,
    file_type: str
) -> ParsedDocument:
    """단순 텍스트 파싱 (문단 단위)"""

    paragraphs = re.split(r"\n\s*\n", text)
    blocks = []

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        blocks.append(ContentBlock(
            text=p,
            block_type="paragraph"
        ))

    return ParsedDocument(
        text=text,
        blocks=blocks,
        metadata={
            "file_name": file_name,
            "file_type": file_type,
            "title": extract_title(text)
        }
    )


# ─────────────────────────────────────────────────────────────
# 법률 / SOP 조항 파서
# ─────────────────────────────────────────────────────────────

ARTICLE_PATTERNS = [
    (r'^제\s*(\d+)\s*조', 'article'),
    (r'^제\s*(\d+)\s*장', 'chapter'),
    (r'^제\s*(\d+)\s*절', 'section'),
    (r'^(\d+\.\d+)', 'subsection'),
    (r'^(\d+)\.', 'item'),
]

def parse_articles(
    text: str,
    file_name: str,
    file_type: str
) -> ParsedDocument:
    """법률/SOP 조항 단위 파싱"""

    lines = text.split("\n")
    blocks: List[ContentBlock] = []

    current_lines = []
    current_meta = {
        "article_num": None,
        "article_type": "intro"
    }

    def flush():
        if current_lines:
            blocks.append(ContentBlock(
                text="\n".join(current_lines).strip(),
                block_type=current_meta["article_type"],
                section=current_meta["article_num"],
                metadata=current_meta.copy()
            ))

    for line in lines:
        line_strip = line.strip()
        matched = False

        for pattern, a_type in ARTICLE_PATTERNS:
            m = re.match(pattern, line_strip)
            if m:
                flush()
                current_lines.clear()
                current_meta = {
                    "article_num": m.group(1),
                    "article_type": a_type
                }
                matched = True
                break

        current_lines.append(line)

    flush()

    return ParsedDocument(
        text=text,
        blocks=[b for b in blocks if b.text],
        metadata={
            "file_name": file_name,
            "file_type": file_type,
            "title": extract_title(text),
            "doc_type": "article"
        }
    )
