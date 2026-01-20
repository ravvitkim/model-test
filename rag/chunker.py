"""
텍스트 청킹 - 다양한 분할 전략 지원
- sentence: 문장 단위
- paragraph: 문단 단위  
- article: 조항 단위 (SOP/법률 문서용)
- recursive: RecursiveCharacterTextSplitter (랭체인 스타일)  ← NEW
- semantic: 의미 기반 분할 (임베딩 유사도)                    ← NEW
- llm: LLM 기반 구조 파싱                                    ← NEW
"""

import re
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
from .parser import ParsedDocument

@dataclass
class Chunk:
    """청크 데이터"""
    text: str
    index: int
    metadata: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# 조항 패턴 정의 (SOP, 법률 문서용)
# ═══════════════════════════════════════════════════════════════════════════

ARTICLE_PATTERNS = [
    (r'^제\s*(\d+)\s*조', 'article'),
    (r'^제\s*(\d+)\s*장', 'chapter'),
    (r'^제\s*(\d+)\s*절', 'section'),
    (r'^(\d+\.\d+\.\d+)', 'subsection'),
    (r'^(\d+\.\d+)', 'subsection'),
    (r'^(\d+)\.(?:\s|$)', 'item'),
    (r'^([가-힣])\.(?:\s|$)', 'subitem'),
    (r'^\((\d+)\)', 'subitem'),
    (r'^\(([가-힣])\)', 'subitem'),
    (r'^([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮])', 'subitem'),
]


def detect_article_type(line: str) -> Optional[Tuple[str, str]]:
    """라인이 조항 시작인지 감지"""
    line = line.strip()
    for pattern, article_type in ARTICLE_PATTERNS:
        match = re.match(pattern, line)
        if match:
            return (match.group(1), article_type)
    return None


def extract_document_title(text: str) -> str:
    """문서 제목 추출"""
    lines = text.strip().split('\n')
    for line in lines[:5]:
        line = line.strip()
        if line.startswith('제목:') or line.startswith('Title:'):
            return line.split(':', 1)[1].strip()
        if re.match(r'^SOP[-_]?\d+', line, re.IGNORECASE):
            return line
        if line and len(line) > 3 and not line.startswith('#'):
            return line[:100]
    return "제목 없음"


# ═══════════════════════════════════════════════════════════════════════════
# 1. 문장 단위 청킹 (기존)
# ═══════════════════════════════════════════════════════════════════════════

def split_by_sentences(text: str, max_length: int = 300, overlap: int = 50) -> List[str]:
    """문장 단위로 분할 후 max_length 이내로 묶기"""
    sentences = re.split(r'(?<=[.!?。])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        if current_length + sentence_length > max_length and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # 오버랩 처리
            overlap_text = " ".join(current_chunk)
            if len(overlap_text) > overlap:
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
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 2. 문단 단위 청킹 (기존)
# ═══════════════════════════════════════════════════════════════════════════

def split_by_paragraphs(text: str, max_length: int = 1000) -> List[str]:
    """문단 단위로 분할"""
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para)
        
        if para_length > max_length:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0
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


# ═══════════════════════════════════════════════════════════════════════════
# 3. 조항 단위 청킹 (기존)
# ═══════════════════════════════════════════════════════════════════════════

def split_by_articles(text: str) -> List[dict]:
    """텍스트를 조항 단위로 분할"""
    lines = text.split('\n')
    articles = []
    current_article = {
        'lines': [],
        'article_num': None,
        'article_type': 'intro',
    }
    
    for line in lines:
        detected = detect_article_type(line)
        
        if detected:
            if current_article['lines']:
                articles.append({
                    'text': '\n'.join(current_article['lines']).strip(),
                    'article_num': current_article['article_num'],
                    'article_type': current_article['article_type'],
                })
            
            current_article = {
                'lines': [line],
                'article_num': detected[0],
                'article_type': detected[1],
            }
        else:
            current_article['lines'].append(line)
    
    if current_article['lines']:
        articles.append({
            'text': '\n'.join(current_article['lines']).strip(),
            'article_num': current_article['article_num'],
            'article_type': current_article['article_type'],
        })
    
    return [a for a in articles if a['text'].strip()]


def split_long_article(text: str, max_length: int, overlap: int = 50) -> List[str]:
    """긴 조항을 max_length 기준으로 분할"""
    if len(text) <= max_length:
        return [text]
    return split_by_sentences(text, max_length=max_length, overlap=overlap)


def chunk_by_articles(text: str, max_length: int = 300, overlap: int = 50) -> List[dict]:
    """조항 단위 청킹"""
    doc_title = extract_document_title(text)
    articles = split_by_articles(text)
    
    chunks = []
    current_section = None
    
    for article in articles:
        if article['article_type'] in ('chapter', 'section'):
            current_section = f"{article['article_type']}_{article['article_num']}"
        
        if len(article['text']) > max_length:
            sub_texts = split_long_article(article['text'], max_length, overlap)
            for i, sub_text in enumerate(sub_texts):
                chunks.append({
                    'text': sub_text,
                    'article_num': article['article_num'],
                    'article_type': article['article_type'],
                    'section': current_section,
                    'doc_title': doc_title,
                    'chunk_part': f"{i+1}/{len(sub_texts)}" if len(sub_texts) > 1 else None,
                })
        else:
            chunks.append({
                'text': article['text'],
                'article_num': article['article_num'],
                'article_type': article['article_type'],
                'section': current_section,
                'doc_title': doc_title,
                'chunk_part': None,
            })
    
    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 4. RecursiveCharacterTextSplitter (랭체인 스타일) ← NEW
# ═══════════════════════════════════════════════════════════════════════════

class RecursiveCharacterTextSplitter:
    """
    LangChain 스타일 RecursiveCharacterTextSplitter
    - 여러 구분자를 순차적으로 시도하여 청크 분할
    - chunk_size 내에서 최대한 의미 있는 단위로 분할
    """
    
    DEFAULT_SEPARATORS = [
        "\n\n",      # 문단
        "\n",        # 줄바꿈
        "。",        # 한국어/일본어 마침표
        ".",         # 영어 마침표
        "!",
        "?",
        ";",
        ":",
        ",",
        " ",         # 공백
        "",          # 문자 단위 (최후 수단)
    ]
    
    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        separators: List[str] = None,
        keep_separator: bool = True,
        length_function: Callable[[str], int] = len,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.keep_separator = keep_separator
        self.length_function = length_function
    
    def split_text(self, text: str) -> List[str]:
        """텍스트를 재귀적으로 분할"""
        return self._split_text(text, self.separators)
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """재귀적 분할 구현"""
        final_chunks = []
        
        # 현재 사용할 구분자 찾기
        separator = separators[-1]  # 기본값: 마지막 구분자
        new_separators = []
        
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break
        
        # 구분자로 텍스트 분할
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)  # 문자 단위 분할
        
        # 청크 병합
        good_splits = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_text = split
            if self.keep_separator and separator:
                split_text = split + separator if split != splits[-1] else split
            
            split_length = self.length_function(split_text)
            
            # 단일 분할이 chunk_size 초과 시 더 작은 구분자로 재귀 분할
            if split_length > self.chunk_size:
                if current_chunk:
                    merged = self._merge_splits(current_chunk, separator)
                    final_chunks.extend(merged)
                    current_chunk = []
                    current_length = 0
                
                if new_separators:
                    sub_chunks = self._split_text(split_text, new_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(split_text[:self.chunk_size])
            
            elif current_length + split_length > self.chunk_size:
                if current_chunk:
                    merged = self._merge_splits(current_chunk, separator)
                    final_chunks.extend(merged)
                
                # 오버랩 처리
                current_chunk = self._get_overlap_splits(current_chunk, separator)
                current_length = sum(self.length_function(s) for s in current_chunk)
                current_chunk.append(split_text)
                current_length += split_length
            else:
                current_chunk.append(split_text)
                current_length += split_length
        
        if current_chunk:
            merged = self._merge_splits(current_chunk, separator)
            final_chunks.extend(merged)
        
        return [c.strip() for c in final_chunks if c.strip()]
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """분할된 텍스트 병합"""
        if not splits:
            return []
        
        merged = separator.join(splits) if separator else "".join(splits)
        return [merged] if merged.strip() else []
    
    def _get_overlap_splits(self, splits: List[str], separator: str) -> List[str]:
        """오버랩을 위한 마지막 분할들 반환"""
        if not splits or self.chunk_overlap <= 0:
            return []
        
        overlap_splits = []
        current_length = 0
        
        for split in reversed(splits):
            split_length = self.length_function(split)
            if current_length + split_length <= self.chunk_overlap:
                overlap_splits.insert(0, split)
                current_length += split_length
            else:
                break
        
        return overlap_splits


def split_recursive(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """RecursiveCharacterTextSplitter 간편 함수"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)


# ═══════════════════════════════════════════════════════════════════════════
# 5. SemanticSplitter (의미 기반 분할) ← NEW
# ═══════════════════════════════════════════════════════════════════════════

class SemanticSplitter:
    """
    의미 기반 텍스트 분할
    - 문장 간 임베딩 유사도를 계산하여 의미적 경계에서 분할
    - 유사도가 급격히 떨어지는 지점 = 주제 전환점
    """
    
    def __init__(
        self,
        embed_function: Callable[[str], np.ndarray] = None,
        threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
    ):
        self.embed_function = embed_function
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def split_text(self, text: str) -> List[str]:
        """의미 기반 분할"""
        # 문장 단위로 먼저 분할
        sentences = self._split_to_sentences(text)
        
        if len(sentences) <= 1:
            return sentences
        
        if self.embed_function is None:
            # 임베딩 함수 없으면 문장 단위 반환
            return self._merge_to_chunks(sentences)
        
        # 각 문장 임베딩
        embeddings = [self.embed_function(s) for s in sentences]
        
        # 연속 문장 간 유사도 계산
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        # 분할 지점 찾기 (유사도가 threshold 이하인 지점)
        split_points = [0]
        for i, sim in enumerate(similarities):
            if sim < self.threshold:
                split_points.append(i + 1)
        split_points.append(len(sentences))
        
        # 청크 생성
        chunks = []
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            chunk_text = " ".join(sentences[start:end])
            
            # max_chunk_size 초과 시 추가 분할
            if len(chunk_text) > self.max_chunk_size:
                sub_chunks = split_by_sentences(chunk_text, self.max_chunk_size)
                chunks.extend(sub_chunks)
            elif len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
            elif chunks:
                # 너무 짧으면 이전 청크에 병합
                chunks[-1] += " " + chunk_text
            else:
                chunks.append(chunk_text)
        
        return chunks
    
    def _split_to_sentences(self, text: str) -> List[str]:
        """문장 분할"""
        sentences = re.split(r'(?<=[.!?。])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _merge_to_chunks(self, sentences: List[str]) -> List[str]:
        """문장들을 max_chunk_size 이내로 병합"""
        chunks = []
        current = []
        current_len = 0
        
        for sent in sentences:
            if current_len + len(sent) > self.max_chunk_size and current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            current.append(sent)
            current_len += len(sent)
        
        if current:
            chunks.append(" ".join(current))
        
        return chunks
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """코사인 유사도 계산"""
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def split_semantic(
    text: str, 
    embed_function: Callable = None,
    threshold: float = 0.5,
    max_chunk_size: int = 300
) -> List[str]:
    """SemanticSplitter 간편 함수"""
    splitter = SemanticSplitter(
        embed_function=embed_function,
        threshold=threshold,
        max_chunk_size=max_chunk_size
    )
    return splitter.split_text(text)


# ═══════════════════════════════════════════════════════════════════════════
# 6. LLM 기반 파싱 ← NEW
# ═══════════════════════════════════════════════════════════════════════════

LLM_PARSING_PROMPT = """다음 문서의 논리적 구조를 분석하고, 의미 있는 단위로 분할해주세요.

각 섹션의 시작과 끝 위치를 JSON 배열로 반환해주세요.
형식: [{"title": "섹션제목", "start": 시작인덱스, "end": 끝인덱스}, ...]

문서:
{text}

JSON 응답만 반환하세요:"""


def split_by_llm(
    text: str,
    llm_function: Callable[[str], str] = None,
    max_chunk_size: int = 300,
    fallback_method: str = "recursive"
) -> List[dict]:
    """
    LLM 기반 문서 구조 파싱
    
    Args:
        text: 원본 텍스트
        llm_function: LLM 호출 함수 (prompt -> response)
        max_chunk_size: 최대 청크 크기
        fallback_method: LLM 실패 시 대체 방법
    
    Returns:
        [{'text': str, 'title': str, ...}, ...]
    """
    if llm_function is None:
        # LLM 함수 없으면 fallback
        return _fallback_split(text, max_chunk_size, fallback_method)
    
    # 텍스트가 너무 길면 앞부분만 분석
    analysis_text = text[:8000] if len(text) > 8000 else text
    
    try:
        prompt = LLM_PARSING_PROMPT.format(text=analysis_text)
        response = llm_function(prompt)
        
        # JSON 파싱
        import json
        
        # JSON 부분 추출
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            raise ValueError("JSON not found in response")
        
        sections = json.loads(json_match.group())
        
        # 섹션별 텍스트 추출
        chunks = []
        for i, section in enumerate(sections):
            start = section.get('start', 0)
            end = section.get('end', len(text))
            section_text = text[start:end].strip()
            
            if len(section_text) > max_chunk_size:
                # 긴 섹션은 추가 분할
                sub_chunks = split_recursive(section_text, max_chunk_size)
                for j, sub in enumerate(sub_chunks):
                    chunks.append({
                        'text': sub,
                        'title': section.get('title', f'Section {i+1}'),
                        'section_index': i,
                        'sub_index': j,
                        'parse_method': 'llm'
                    })
            else:
                chunks.append({
                    'text': section_text,
                    'title': section.get('title', f'Section {i+1}'),
                    'section_index': i,
                    'parse_method': 'llm'
                })
        
        return chunks if chunks else _fallback_split(text, max_chunk_size, fallback_method)
    
    except Exception as e:
        print(f"⚠️ LLM 파싱 실패, fallback 사용: {e}")
        return _fallback_split(text, max_chunk_size, fallback_method)


def _fallback_split(text: str, max_chunk_size: int, method: str) -> List[dict]:
    """LLM 파싱 실패 시 대체 분할"""
    if method == "recursive":
        raw_chunks = split_recursive(text, max_chunk_size)
    elif method == "sentence":
        raw_chunks = split_by_sentences(text, max_chunk_size)
    else:
        raw_chunks = split_by_paragraphs(text, max_chunk_size)
    
    return [{'text': c, 'parse_method': f'fallback_{method}'} for c in raw_chunks]


# ═══════════════════════════════════════════════════════════════════════════
# 통합 청킹 함수
# ═══════════════════════════════════════════════════════════════════════════

CHUNK_METHODS = {
    "sentence": "문장 단위 분할",
    "paragraph": "문단 단위 분할",
    "article": "조항 단위 분할 (SOP/법률)",
    "recursive": "RecursiveCharacterTextSplitter (랭체인)",
    "semantic": "의미 기반 분할 (임베딩 유사도)",
    "llm": "LLM 기반 구조 파싱",
}


def create_chunks(
    text: str, 
    chunk_size: int = 300, 
    overlap: int = 50,
    method: str = "sentence",
    embed_function: Callable = None,
    llm_function: Callable = None,
    semantic_threshold: float = 0.5,
) -> List[Chunk]:
    """
    텍스트를 청크로 분할 (통합 인터페이스)
    
    Args:
        text: 원본 텍스트
        chunk_size: 청크 크기
        overlap: 중복 크기
        method: "sentence", "paragraph", "article", "recursive", "semantic", "llm"
        embed_function: semantic 분할용 임베딩 함수
        llm_function: llm 파싱용 LLM 함수
        semantic_threshold: semantic 분할 임계값
    """
    
    if method == "article":
        # 조항 단위 청킹
        raw_chunks = chunk_by_articles(text, max_length=chunk_size, overlap=overlap)
        chunks = []
        for i, chunk_data in enumerate(raw_chunks):
            chunks.append(Chunk(
                text=chunk_data['text'].strip(),
                index=i,
                metadata={
                    "chunk_index": i,
                    "total_chunks": len(raw_chunks),
                    "chunk_method": "article",
                    "article_num": chunk_data.get('article_num'),
                    "article_type": chunk_data.get('article_type'),
                    "section": chunk_data.get('section'),
                    "doc_title": chunk_data.get('doc_title'),
                    "chunk_part": chunk_data.get('chunk_part'),
                }
            ))
        return chunks
    
    elif method == "recursive":
        # RecursiveCharacterTextSplitter
        raw_chunks = split_recursive(text, chunk_size, overlap)
    
    elif method == "semantic":
        # 의미 기반 분할
        raw_chunks = split_semantic(
            text, 
            embed_function=embed_function,
            threshold=semantic_threshold,
            max_chunk_size=chunk_size
        )
    
    elif method == "llm":
        # LLM 기반 파싱
        raw_data = split_by_llm(
            text,
            llm_function=llm_function,
            max_chunk_size=chunk_size,
            fallback_method="recursive"
        )
        chunks = []
        for i, data in enumerate(raw_data):
            chunks.append(Chunk(
                text=data['text'].strip(),
                index=i,
                metadata={
                    "chunk_index": i,
                    "total_chunks": len(raw_data),
                    "chunk_method": "llm",
                    "title": data.get('title'),
                    "section_index": data.get('section_index'),
                    "parse_method": data.get('parse_method'),
                }
            ))
        return chunks
    
    elif method == "paragraph":
        raw_chunks = split_by_paragraphs(text, max_length=chunk_size)
    
    else:  # sentence (기본값)
        raw_chunks = split_by_sentences(text, max_length=chunk_size, overlap=overlap)
    
    # 공통 Chunk 객체 생성
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        if chunk_text.strip():
            chunks.append(Chunk(
                text=chunk_text.strip(),
                index=i,
                metadata={
                    "chunk_index": i, 
                    "total_chunks": len(raw_chunks),
                    "chunk_method": method,
                }
            ))
    
    return chunks

def create_chunks_from_blocks(
    doc: ParsedDocument,
    chunk_size: int = 300,
    overlap: int = 50,
    method: str = "recursive"
) -> List[Chunk]:
    """
    block 기반 청킹 (메타데이터 유지)
    """
    chunks: List[Chunk] = []
    idx = 0

    for block in doc.blocks:
        if method == "recursive":
            texts = split_recursive(block.text, chunk_size, overlap)
        else:
            texts = [block.text]

        for t in texts:
            if not t.strip():
                continue

            chunks.append(Chunk(
                text=t.strip(),
                index=idx,
                metadata={
                    **doc.metadata,
                    **block.metadata,
                    "block_type": block.block_type,
                    "page": block.page,
                    "section": block.section,
                    "chunk_method": f"block_{method}"
                }
            ))
            idx += 1

    return chunks



def get_available_methods() -> dict:
    """사용 가능한 청킹 방법 목록"""
    return CHUNK_METHODS.copy()