"""
문서 로더 v6.0 - Docling 기반
- PDF, DOCX, HTML, 이미지 등 다양한 형식 지원
- 표(Table) 파싱 지원
- 구조화된 메타데이터 추출
"""

import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import io
import tempfile
import os


@dataclass
class ContentBlock:
    """문서의 의미 단위 블록"""
    text: str
    block_type: str  # title, paragraph, table, list, article 등
    level: int = 0
    page: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """파싱된 문서"""
    text: str
    blocks: List[ContentBlock]
    metadata: Dict
    tables: List[Dict] = field(default_factory=list)  # 표 데이터


def get_supported_extensions() -> list:
    """지원되는 파일 확장자"""
    return [".txt", ".md", ".pdf", ".docx", ".doc", ".html", ".htm", ".csv", ".xlsx", ".pptx", ".png", ".jpg", ".jpeg"]


def load_document(filename: str, content: bytes) -> ParsedDocument:
    """
    문서 로드 및 파싱 (메인 진입점)
    
    Returns:
        ParsedDocument: 파싱된 문서 (text + blocks + metadata + tables)
    """
    ext = Path(filename).suffix.lower()

    # DOCX: 하이브리드 방식 (Docling 표 + python-docx 텍스트)
    if ext in [".docx", ".doc"]:
        return _load_docx_hybrid(filename, content)

    # PDF, PPTX, XLSX, HTML, 이미지 → Docling 시도
    if ext in [".pdf", ".pptx", ".xlsx", ".html", ".htm", ".png", ".jpg", ".jpeg"]:
        try:
            return _load_with_docling(filename, content, ext)
        except ImportError:
            print("⚠️ Docling not installed, falling back to basic parser")
            if ext == ".pdf":
                return _load_pdf_basic(filename, content)
            elif ext in [".html", ".htm"]:
                return _load_html_basic(filename, content)
        except Exception as e:
            print(f"⚠️ Docling failed: {e}, falling back to basic parser")
            if ext == ".pdf":
                return _load_pdf_basic(filename, content)
            elif ext in [".html", ".htm"]:
                return _load_html_basic(filename, content)

    # 텍스트 기반 파일
    if ext in [".txt", ".md"]:
        text = _decode_text(content)
        if _is_article_document(text):
            return _parse_articles(text, filename, ext)
        return _parse_plain_text(text, filename, ext)

    if ext == ".csv":
        return _load_csv(filename, content)

    # 기본 텍스트 처리
    text = _decode_text(content)
    return _parse_plain_text(text, filename, ext)


def _load_docx_hybrid(filename: str, content: bytes) -> ParsedDocument:
    """
    DOCX 하이브리드 파싱
    - 문서 순서대로 텍스트 추출 (중요!)
    - 표(Table) 파싱 지원
    """
    tables_data = []
    
    # python-docx로 문서 순서대로 파싱
    try:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError:
        return ParsedDocument(
            text="python-docx가 설치되지 않았습니다.",
            blocks=[],
            metadata={"file_name": filename, "error": "python-docx not installed"},
            tables=[]
        )
    
    doc = Document(io.BytesIO(content))
    all_text = []
    
    # 문서 순서대로 순회 (핵심!)
    for element in doc.element.body:
        # 단락(Paragraph)
        if element.tag.endswith('p'):
            para = Paragraph(element, doc)
            text = para.text.strip()
            if text:
                all_text.append(text)
        
        # 표(Table)
        elif element.tag.endswith('tbl'):
            table = Table(element, doc)
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append(cells)
                # 표 내용도 텍스트에 포함 (SOP ID 추적용)
                all_text.append(' | '.join(cells))
            
            if rows:
                tables_data.append({"rows": rows, "source": "python-docx"})
    
    full_text = '\n'.join(all_text)
    
    # 조항 단위 블록 생성
    blocks = _extract_article_blocks(full_text)
    
    # 메타데이터
    metadata = {
        "file_name": filename,
        "file_type": "docx",
        "title": _extract_title(full_text),
        "table_count": len(tables_data),
        "parser": "python-docx (ordered)"
    }
    metadata.update(_extract_sop_metadata(full_text))
    
    return ParsedDocument(
        text=full_text,
        blocks=blocks,
        metadata=metadata,
        tables=tables_data
    )


# ═══════════════════════════════════════════════════════════════════════════
# Docling 기반 파서 (핵심!)
# ═══════════════════════════════════════════════════════════════════════════

def _load_with_docling(filename: str, content: bytes, ext: str) -> ParsedDocument:
    """Docling을 사용한 고급 문서 파싱"""
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Docling 컨버터 설정
        converter = DocumentConverter()

        # 문서 변환
        result = converter.convert(tmp_path)
        doc = result.document

        # 전체 텍스트
        full_text = doc.export_to_markdown()

        # 블록 추출
        blocks = []
        tables = []

        for item in doc.iterate_items():
            element = item[1] if isinstance(item, tuple) else item

            # 텍스트 추출
            if hasattr(element, 'text'):
                text = element.text
            elif hasattr(element, 'export_to_markdown'):
                text = element.export_to_markdown()
            else:
                continue

            if not text or not text.strip():
                continue

            # 블록 타입 결정
            block_type = "paragraph"
            element_type = type(element).__name__.lower()

            if "title" in element_type or "heading" in element_type:
                block_type = "title"
            elif "table" in element_type:
                block_type = "table"
                # 표 데이터 추출
                table_data = _extract_table_data(element)
                if table_data:
                    tables.append(table_data)
            elif "list" in element_type:
                block_type = "list"

            # 페이지 번호
            page_num = None
            if hasattr(element, 'prov') and element.prov:
                for prov in element.prov:
                    if hasattr(prov, 'page_no'):
                        page_num = prov.page_no
                        break

            blocks.append(ContentBlock(
                text=text.strip(),
                block_type=block_type,
                page=page_num,
                metadata={"source": "docling"}
            ))

        # 조항 패턴 감지 및 재파싱
        if _is_article_document(full_text):
            article_blocks = _extract_article_blocks(full_text)
            if article_blocks:
                blocks = article_blocks

        # 메타데이터 추출
        metadata = {
            "file_name": filename,
            "file_type": ext,
            "title": _extract_title(full_text),
            "total_pages": _count_pages(doc),
            "table_count": len(tables),
            "parser": "docling"
        }

        # SOP 메타데이터 추출
        sop_meta = _extract_sop_metadata(full_text)
        metadata.update(sop_meta)

        return ParsedDocument(
            text=full_text,
            blocks=blocks,
            metadata=metadata,
            tables=tables
        )

    finally:
        # 임시 파일 삭제
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _extract_table_data(element) -> Optional[Dict]:
    """표 요소에서 데이터 추출"""
    try:
        if hasattr(element, 'export_to_dataframe'):
            df = element.export_to_dataframe()
            return {
                "headers": list(df.columns),
                "rows": df.values.tolist(),
                "markdown": element.export_to_markdown() if hasattr(element, 'export_to_markdown') else str(df)
            }
        elif hasattr(element, 'data'):
            data = element.data
            if hasattr(data, 'grid'):
                return {
                    "grid": data.grid,
                    "markdown": element.export_to_markdown() if hasattr(element, 'export_to_markdown') else ""
                }
    except Exception as e:
        print(f"표 추출 실패: {e}")
    return None


def _count_pages(doc) -> int:
    """문서 페이지 수"""
    try:
        if hasattr(doc, 'pages'):
            return len(doc.pages)
    except Exception:
        pass
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# 조항 파싱 (SOP/법률 문서)
# ═══════════════════════════════════════════════════════════════════════════

ARTICLE_PATTERNS = [
    (r'^제\s*(\d+)\s*조', 'article'),
    (r'^제\s*(\d+)\s*장', 'chapter'),
    (r'^제\s*(\d+)\s*절', 'section'),
    (r'^(\d+)\.\s+([가-힣]+)', 'section'),      # "1. 목적", "6. 절차" 형식
    (r'^(\d+\.\d+)\s+([가-힣]+)', 'subsection'), # "6.1 사전 준비", "6.2 시약확인" 형식
]


def _is_article_document(text: str) -> bool:
    """조항 기반 문서인지 감지"""
    patterns = [
        r'제\s*\d+\s*조',
        r'제\s*\d+\s*장',
        r'제\s*\d+\s*절',
        r'^\d+\.\d+',
        r'^SOP[-_]?\d+',
    ]

    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        count += len(matches)

    return count >= 3


def _extract_article_blocks(text: str) -> List[ContentBlock]:
    """조항 단위 블록 추출 (SOP 경계 감지)"""
    lines = text.split('\n')
    blocks = []
    current_lines = []
    current_sop_id = ""  # 현재 SOP ID
    current_meta = {"article_num": None, "article_type": "intro", "title": ""}
    
    sop_id_pattern = re.compile(r'(SOP[-_][A-Z]+[-_]\d+)', re.IGNORECASE)

    def flush():
        if current_lines:
            block_text = '\n'.join(current_lines).strip()
            if block_text:
                blocks.append(ContentBlock(
                    text=block_text,
                    block_type=current_meta["article_type"],
                    section=current_meta["article_num"],
                    metadata={
                        "article_num": current_meta["article_num"],
                        "article_type": current_meta["article_type"],
                        "title": current_meta.get("title", ""),
                        "sop_id": current_sop_id
                    }
                ))

    for line in lines:
        line_strip = line.strip()
        
        # SOP ID 추출 - 새 SOP 시작이면 현재 블록 flush 먼저!
        sop_match = sop_id_pattern.search(line_strip)
        if sop_match:
            new_sop_id = sop_match.group(1).upper().replace('_', '-')
            if new_sop_id != current_sop_id:
                # 새 SOP 시작 → 현재 블록 저장 후 SOP ID 갱신
                flush()
                current_lines = []
                current_meta = {"article_num": None, "article_type": "intro", "title": ""}
                current_sop_id = new_sop_id
        
        # 조항 패턴 매칭
        matched = False
        for pattern, a_type in ARTICLE_PATTERNS:
            m = re.match(pattern, line_strip)
            if m:
                flush()
                current_lines = [line]
                title = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else ""
                current_meta = {
                    "article_num": m.group(1),
                    "article_type": a_type,
                    "title": title
                }
                matched = True
                break

        if not matched:
            current_lines.append(line)

    flush()
    return blocks


def _parse_articles(text: str, filename: str, ext: str) -> ParsedDocument:
    """조항 단위 파싱"""
    blocks = _extract_article_blocks(text)
    metadata = {
        "file_name": filename,
        "file_type": ext,
        "title": _extract_title(text),
        "parser": "article"
    }
    metadata.update(_extract_sop_metadata(text))

    return ParsedDocument(text=text, blocks=blocks, metadata=metadata)


def _parse_plain_text(text: str, filename: str, ext: str) -> ParsedDocument:
    """단순 텍스트 파싱"""
    paragraphs = re.split(r'\n\s*\n', text)
    blocks = []

    for p in paragraphs:
        p = p.strip()
        if p:
            blocks.append(ContentBlock(text=p, block_type="paragraph"))

    return ParsedDocument(
        text=text,
        blocks=blocks,
        metadata={
            "file_name": filename,
            "file_type": ext,
            "title": _extract_title(text),
            "parser": "plain"
        }
    )


# ═══════════════════════════════════════════════════════════════════════════
# 메타데이터 추출
# ═══════════════════════════════════════════════════════════════════════════

def _extract_title(text: str) -> str:
    """문서 제목 추출"""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines[:10]:
        if line.lower().startswith(("title:", "제목:")):
            return line.split(':', 1)[1].strip()
        if re.match(r'^SOP[-_]?\d+', line, re.IGNORECASE):
            return line[:100]
        if len(line) > 5 and not line.startswith('#'):
            return line[:100]
    return "제목 없음"


def _extract_sop_metadata(text: str) -> Dict:
    """SOP 관련 메타데이터 추출"""
    metadata = {}

    # SOP ID
    sop_match = re.search(r'(SOP[-_]?[A-Z]*[-_]?\d+)', text, re.IGNORECASE)
    if sop_match:
        metadata["sop_id"] = sop_match.group(1)

    # 버전
    ver_match = re.search(r'(?:Version|Ver|버전|개정)[\s.:]*(\d+\.?\d*)', text, re.IGNORECASE)
    if ver_match:
        metadata["version"] = ver_match.group(1)

    # 부서
    dept_match = re.search(r'(?:부서|Dept|Department)[\s:]*([가-힣\w\s]+?)(?:\n|$)', text, re.IGNORECASE)
    if dept_match:
        metadata["dept"] = dept_match.group(1).strip()

    # 시행일
    date_match = re.search(r'(?:시행일|Effective|발효)[\s:]*(\d{4}[-./]\d{1,2}[-./]\d{1,2})', text, re.IGNORECASE)
    if date_match:
        metadata["effective_date"] = date_match.group(1)

    return metadata


# ═══════════════════════════════════════════════════════════════════════════
# Fallback 파서들 (Docling 없을 때)
# ═══════════════════════════════════════════════════════════════════════════

def _decode_text(content: bytes) -> str:
    """바이트 → 텍스트 디코딩"""
    for encoding in ["utf-8", "cp949", "euc-kr", "latin-1"]:
        try:
            return content.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return content.decode("utf-8", errors="ignore")


def _load_pdf_basic(filename: str, content: bytes) -> ParsedDocument:
    """기본 PDF 파싱 (PyMuPDF)"""
    try:
        import fitz
    except ImportError:
        return ParsedDocument(
            text="PDF 파싱 라이브러리(PyMuPDF)가 설치되지 않았습니다.",
            blocks=[],
            metadata={"file_name": filename, "error": "pymupdf not installed"}
        )

    doc = fitz.open(stream=content, filetype="pdf")
    blocks = []
    all_text = []
    tables = []

    for page_idx, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            blocks.append(ContentBlock(
                text=text,
                block_type="page",
                page=page_idx + 1,
                metadata={"source": "pymupdf"}
            ))
            all_text.append(text)

        # 표 추출 시도
        try:
            page_tables = page.find_tables()
            for table in page_tables:
                table_data = {
                    "page": page_idx + 1,
                    "rows": table.extract()
                }
                tables.append(table_data)
        except Exception:
            pass

    full_text = '\n\n'.join(all_text)

    # 조항 재파싱
    if _is_article_document(full_text):
        article_blocks = _extract_article_blocks(full_text)
        if article_blocks:
            blocks = article_blocks

    metadata = {
        "file_name": filename,
        "file_type": "pdf",
        "total_pages": len(doc),
        "title": _extract_title(full_text),
        "table_count": len(tables),
        "parser": "pymupdf"
    }
    metadata.update(_extract_sop_metadata(full_text))

    return ParsedDocument(text=full_text, blocks=blocks, metadata=metadata, tables=tables)


def _load_docx_basic(filename: str, content: bytes) -> ParsedDocument:
    """기본 DOCX 파싱"""
    try:
        from docx import Document
    except ImportError:
        return ParsedDocument(
            text="DOCX 파싱 라이브러리(python-docx)가 설치되지 않았습니다.",
            blocks=[],
            metadata={"file_name": filename, "error": "python-docx not installed"}
        )

    doc = Document(io.BytesIO(content))
    blocks = []
    all_text = []
    tables = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            # 스타일로 타입 결정
            style_name = para.style.name.lower() if para.style else ""
            block_type = "title" if "heading" in style_name else "paragraph"
            blocks.append(ContentBlock(text=text, block_type=block_type))
            all_text.append(text)

    # 표 추출
    for table in doc.tables:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(cells)
            all_text.append(' | '.join(cells))
        if rows:
            tables.append({"rows": rows})
            blocks.append(ContentBlock(
                text='\n'.join([' | '.join(r) for r in rows]),
                block_type="table"
            ))

    full_text = '\n\n'.join(all_text)

    if _is_article_document(full_text):
        article_blocks = _extract_article_blocks(full_text)
        if article_blocks:
            blocks = article_blocks

    metadata = {
        "file_name": filename,
        "file_type": "docx",
        "title": _extract_title(full_text),
        "table_count": len(tables),
        "parser": "python-docx"
    }
    metadata.update(_extract_sop_metadata(full_text))

    return ParsedDocument(text=full_text, blocks=blocks, metadata=metadata, tables=tables)


def _load_html_basic(filename: str, content: bytes) -> ParsedDocument:
    """기본 HTML 파싱"""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        text = _decode_text(content)
        return _parse_plain_text(text, filename, ".html")

    html = _decode_text(content)
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    blocks = []
    tables = []

    # 표 추출
    for table in soup.find_all('table'):
        rows = []
        for tr in table.find_all('tr'):
            cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            rows.append(cells)
        if rows:
            tables.append({"rows": rows})
            blocks.append(ContentBlock(
                text='\n'.join([' | '.join(r) for r in rows]),
                block_type="table"
            ))
        table.decompose()

    # 나머지 텍스트
    text = soup.get_text('\n', strip=True)
    for para in text.split('\n\n'):
        para = para.strip()
        if para:
            blocks.append(ContentBlock(text=para, block_type="paragraph"))

    full_text = soup.get_text('\n', strip=True)

    return ParsedDocument(
        text=full_text,
        blocks=blocks,
        metadata={
            "file_name": filename,
            "file_type": "html",
            "title": _extract_title(full_text),
            "table_count": len(tables),
            "parser": "beautifulsoup"
        },
        tables=tables
    )


def _load_csv(filename: str, content: bytes) -> ParsedDocument:
    """CSV 파싱"""
    text = _decode_text(content)
    lines = text.splitlines()

    rows = []
    for line in lines:
        cells = [c.strip('" ') for c in line.split(',')]
        rows.append(cells)

    table_text = '\n'.join([' | '.join(row) for row in rows])

    return ParsedDocument(
        text=table_text,
        blocks=[ContentBlock(text=table_text, block_type="table")],
        metadata={
            "file_name": filename,
            "file_type": "csv",
            "title": filename,
            "row_count": len(rows),
            "parser": "csv"
        },
        tables=[{"rows": rows}]
    )