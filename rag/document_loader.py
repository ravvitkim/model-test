"""
문서 로더 - 다양한 파일 형식 지원
"""

import re
from pathlib import Path
from typing import Optional


def get_supported_extensions() -> list:
    """지원되는 파일 확장자 목록"""
    return [".txt", ".md", ".pdf", ".docx", ".doc", ".hwp", ".html", ".csv"]


def load_document(filename: str, content: bytes) -> str:
    """
    파일 내용을 텍스트로 변환
    
    Args:
        filename: 파일명
        content: 파일 바이트 내용
    
    Returns:
        추출된 텍스트
    """
    ext = Path(filename).suffix.lower()
    
    if ext in [".txt", ".md"]:
        return _load_text(content)
    
    elif ext == ".pdf":
        return _load_pdf(content)
    
    elif ext in [".docx", ".doc"]:
        return _load_docx(content)
    
    elif ext == ".hwp":
        return _load_hwp(content)
    
    elif ext == ".html":
        return _load_html(content)
    
    elif ext == ".csv":
        return _load_csv(content)
    
    else:
        # 기본: UTF-8 텍스트로 시도
        try:
            return content.decode("utf-8")
        except:
            return content.decode("cp949", errors="ignore")


def _load_text(content: bytes) -> str:
    """텍스트 파일 로드"""
    try:
        return content.decode("utf-8")
    except:
        return content.decode("cp949", errors="ignore")


def _load_pdf(content: bytes) -> str:
    """PDF 파일 로드"""
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(stream=content, filetype="pdf")
        text_parts = []
        
        for page in doc:
            text_parts.append(page.get_text())
        
        doc.close()
        return "\n".join(text_parts)
    
    except ImportError:
        try:
            # 대체: pdfplumber
            import pdfplumber
            import io
            
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text_parts = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                return "\n".join(text_parts)
        
        except ImportError:
            raise ImportError("PDF 처리를 위해 'pip install pymupdf' 또는 'pip install pdfplumber' 설치 필요")


def _load_docx(content: bytes) -> str:
    """DOCX 파일 로드"""
    try:
        from docx import Document
        import io
        
        doc = Document(io.BytesIO(content))
        text_parts = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
        
        # 테이블 내용도 추출
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    text_parts.append(" | ".join(row_text))
        
        return "\n".join(text_parts)
    
    except ImportError:
        raise ImportError("DOCX 처리를 위해 'pip install python-docx' 설치 필요")


def _load_hwp(content: bytes) -> str:
    """HWP 파일 로드"""
    try:
        import olefile
        import zlib
        import io
        
        ole = olefile.OleFileIO(io.BytesIO(content))
        
        # HWP 본문 스트림
        if ole.exists("BodyText/Section0"):
            encoded = ole.openstream("BodyText/Section0").read()
            
            # 압축 해제
            try:
                decoded = zlib.decompress(encoded, -15)
            except:
                decoded = encoded
            
            # 텍스트 추출 (간단한 방법)
            text = decoded.decode("utf-16-le", errors="ignore")
            
            # 제어 문자 제거
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
            
            ole.close()
            return text
        
        ole.close()
        return ""
    
    except ImportError:
        raise ImportError("HWP 처리를 위해 'pip install olefile' 설치 필요")
    except Exception as e:
        return f"HWP 파일 로드 오류: {str(e)}"


def _load_html(content: bytes) -> str:
    """HTML 파일 로드"""
    try:
        from bs4 import BeautifulSoup
        
        try:
            html_text = content.decode("utf-8")
        except:
            html_text = content.decode("cp949", errors="ignore")
        
        soup = BeautifulSoup(html_text, "html.parser")
        
        # 스크립트, 스타일 태그 제거
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text(separator="\n")
        
        # 연속 공백 정리
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
    
    except ImportError:
        # BeautifulSoup 없으면 간단한 정규식으로
        try:
            html_text = content.decode("utf-8")
        except:
            html_text = content.decode("cp949", errors="ignore")
        
        # 태그 제거
        text = re.sub(r'<[^>]+>', '', html_text)
        return text


def _load_csv(content: bytes) -> str:
    """CSV 파일 로드"""
    try:
        text = content.decode("utf-8")
    except:
        text = content.decode("cp949", errors="ignore")
    
    # CSV를 읽기 쉬운 형태로 변환
    lines = text.strip().split("\n")
    formatted = []
    
    for line in lines:
        # 간단한 CSV 파싱
        cells = line.split(",")
        formatted.append(" | ".join(cell.strip().strip('"') for cell in cells))
    
    return "\n".join(formatted)


def clean_text(text: str) -> str:
    """텍스트 정리"""
    # 연속 공백을 단일 공백으로
    text = re.sub(r' +', ' ', text)
    
    # 연속 줄바꿈을 최대 2개로
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text