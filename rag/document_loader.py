"""
문서 로더 + 파싱 진입점 (수정됨)
- ContentBlock import 추가
- 파일 타입별 최적화 파싱
"""

from pathlib import Path
from typing import Optional
from .parser import (
    ParsedDocument,
    ContentBlock,  # 추가!
    parse_plain_text,
    parse_articles,
)


def get_supported_extensions() -> list:
    return [".txt", ".md", ".pdf", ".docx", ".doc", ".hwp", ".html", ".csv"]


def load_document(filename: str, content: bytes) -> ParsedDocument:
    """
    문서 로드 및 파싱 (메인 진입점)
    
    Returns:
        ParsedDocument: 파싱된 문서 (text + blocks + metadata)
    """
    ext = Path(filename).suffix.lower()

    if ext in [".txt", ".md"]:
        text = _decode_text(content)
        # SOP/법률 패턴 감지 시 조항 파싱
        if _is_article_document(text):
            return parse_articles(text, filename, ext)
        return parse_plain_text(text, filename, ext)

    elif ext == ".pdf":
        return _load_pdf(filename, content)

    elif ext in [".docx", ".doc"]:
        text = _load_docx(content)
        if _is_article_document(text):
            return parse_articles(text, filename, ext)
        return parse_plain_text(text, filename, ext)

    elif ext == ".html":
        text = _load_html(content)
        return parse_plain_text(text, filename, ext)

    elif ext == ".csv":
        text = _load_csv(content)
        return parse_plain_text(text, filename, ext)

    else:
        text = _decode_text(content)
        return parse_plain_text(text, filename, ext)


def _is_article_document(text: str) -> bool:
    """조항 기반 문서인지 감지 (SOP, 법률 등)"""
    import re
    
    # 조항 패턴 카운트
    patterns = [
        r'제\s*\d+\s*조',   # 제1조
        r'제\s*\d+\s*장',   # 제1장
        r'제\s*\d+\s*절',   # 제1절
        r'^\d+\.\d+',       # 1.1
        r'^SOP[-_]?\d+',    # SOP-001
    ]
    
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        count += len(matches)
    
    # 3개 이상의 조항 패턴이 있으면 조항 문서로 판단
    return count >= 3


def _decode_text(content: bytes) -> str:
    """바이트 → 텍스트 디코딩"""
    for encoding in ["utf-8", "cp949", "euc-kr", "latin-1"]:
        try:
            return content.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return content.decode("utf-8", errors="ignore")


def _load_pdf(filename: str, content: bytes) -> ParsedDocument:
    """PDF 파싱 (페이지 메타데이터 포함)"""
    import fitz  # pymupdf

    doc = fitz.open(stream=content, filetype="pdf")
    blocks = []
    all_text = []

    for page_idx, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            blocks.append(
                ContentBlock(
                    text=text,
                    block_type="page",
                    page=page_idx + 1,
                    metadata={"source": "pdf"}
                )
            )
            all_text.append(text)

    full_text = "\n\n".join(all_text)
    
    return ParsedDocument(
        text=full_text,
        blocks=blocks,
        metadata={
            "file_name": filename,
            "file_type": "pdf",
            "total_pages": len(doc),
            "title": _extract_title_from_text(full_text),
        }
    )


def _load_docx(content: bytes) -> str:
    """DOCX 파싱"""
    from docx import Document
    import io

    doc = Document(io.BytesIO(content))
    parts = []

    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text.strip())

    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    return "\n\n".join(parts)


def _load_html(content: bytes) -> str:
    """HTML 파싱"""
    from bs4 import BeautifulSoup

    html = _decode_text(content)
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    lines = [l.strip() for l in soup.get_text("\n").splitlines() if l.strip()]
    return "\n".join(lines)


def _load_csv(content: bytes) -> str:
    """CSV 파싱"""
    text = _decode_text(content)
    lines = text.splitlines()
    return "\n".join(" | ".join(c.strip('" ') for c in l.split(",")) for l in lines)


def _extract_title_from_text(text: str) -> str:
    """텍스트에서 제목 추출"""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:5]:
        if line.lower().startswith(("title:", "제목:")):
            return line.split(":", 1)[1].strip()
        if len(line) > 5:
            return line[:100]
    return "제목 없음"
