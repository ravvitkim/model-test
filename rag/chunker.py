"""
텍스트 청킹 - 문서를 의미 단위로 분할
"""

import re
from typing import List
from dataclasses import dataclass


@dataclass
class Chunk:
    """청크 데이터"""
    text: str
    index: int
    metadata: dict = None


def split_by_sentences(text: str, max_length: int = 500, overlap: int = 50) -> List[str]:
    """
    문장 단위로 분할 후 max_length 이내로 묶기
    
    Args:
        text: 원본 텍스트
        max_length: 청크 최대 길이
        overlap: 청크 간 중복 글자 수
    """
    # 문장 분리 (한국어/영어)
    sentences = re.split(r'(?<=[.!?。])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        if current_length + sentence_length > max_length and current_chunk:
            # 현재 청크 저장
            chunks.append(" ".join(current_chunk))
            
            # 오버랩 처리 - 마지막 몇 문장 유지
            overlap_text = " ".join(current_chunk)
            if len(overlap_text) > overlap:
                # 뒤에서부터 overlap 길이만큼 유지
                overlap_start = len(overlap_text) - overlap
                overlap_sentences = []
                temp_length = 0
                for s in reversed(current_chunk):
                    if temp_length + len(s) <= overlap:
                        overlap_sentences.insert(0, s)
                        temp_length += len(s)
                    else:
                        break
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk = []
                current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # 마지막 청크
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def split_by_paragraphs(text: str, max_length: int = 1000) -> List[str]:
    """문단 단위로 분할"""
    # 빈 줄 기준 분할
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para)
        
        # 문단 자체가 max_length보다 길면 문장 단위로 분할
        if para_length > max_length:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # 긴 문단은 문장 단위로 분할
            sub_chunks = split_by_sentences(para, max_length=max_length)
            chunks.extend(sub_chunks)
        
        elif current_length + para_length > max_length:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_length = para_length
        
        else:
            current_chunk.append(para)
            current_length += para_length
    
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks


def create_chunks(
    text: str, 
    chunk_size: int = 500, 
    overlap: int = 50,
    method: str = "sentence"
) -> List[Chunk]:
    """
    텍스트를 청크로 분할
    
    Args:
        text: 원본 텍스트
        chunk_size: 청크 크기
        overlap: 중복 크기
        method: "sentence" 또는 "paragraph"
    """
    if method == "paragraph":
        raw_chunks = split_by_paragraphs(text, max_length=chunk_size)
    else:
        raw_chunks = split_by_sentences(text, max_length=chunk_size, overlap=overlap)
    
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        if chunk_text.strip():
            chunks.append(Chunk(
                text=chunk_text.strip(),
                index=i,
                metadata={"chunk_index": i, "total_chunks": len(raw_chunks)}
            ))
    
    return chunks