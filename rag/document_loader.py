"""
문서 로더 - PDF, DOCX 텍스트 추출
"""

import io
from typing import List
from pathlib import Path


def load_pdf(file_content: bytes) -> str:
    """PDF에서 텍스트 추출"""
    try:
        from pypdf import PdfReader
        
        reader = PdfReader(io.BytesIO(file_content))
        text_parts = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    except Exception as e:
        raise ValueError(f"PDF 파싱 실패: {str(e)}")


def load_docx(file_content: bytes) -> str:
    """DOCX에서 텍스트 추출"""
    try:
        from docx import Document
        
        doc = Document(io.BytesIO(file_content))
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
        
        return "\n\n".join(text_parts)
    
    except Exception as e:
        raise ValueError(f"DOCX 파싱 실패: {str(e)}")


def load_txt(file_content: bytes) -> str:
    """TXT 파일 로드"""
    try:
        # UTF-8 먼저 시도, 실패하면 CP949 (한글)
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            return file_content.decode('cp949')
    except Exception as e:
        raise ValueError(f"TXT 파싱 실패: {str(e)}")


def load_document(filename: str, file_content: bytes) -> str:
    """파일 확장자에 따라 적절한 로더 선택"""
    ext = Path(filename).suffix.lower()
    
    if ext == '.pdf':
        return load_pdf(file_content)
    elif ext in ['.docx', '.doc']:
        return load_docx(file_content)
    elif ext == '.txt':
        return load_txt(file_content)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}")


def get_supported_extensions() -> List[str]:
    """지원하는 파일 확장자 목록"""
    return ['.pdf', '.docx', '.doc', '.txt']