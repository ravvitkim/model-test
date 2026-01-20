"""
ChromaDB 벡터 스토어 - 리팩토링 v5.1
- 유사도 threshold 추가 (낮은 품질 결과 필터링)
- 검색 품질 지표 추가
- 코드 구조 개선
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import hashlib
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════
# 설정 상수
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_COLLECTION = "documents"
CHROMA_PATH = "./chroma_db"

# 검색 품질 설정
DEFAULT_SIMILARITY_THRESHOLD = 0.35  # 이 이하는 "관련 없음"으로 판단
HIGH_CONFIDENCE_THRESHOLD = 0.65     # 이 이상은 "신뢰도 높음"
MIN_RESULTS_BEFORE_FILTER = 3        # 필터링 전 최소 결과 수

# 임베딩 모델 필터링 기준
MAX_EMBEDDING_DIM = 1024
MAX_MEMORY_MB = 1300

# 전역 캐시
_client: Optional[chromadb.PersistentClient] = None
_embed_models: Dict = {}
_device: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
# 임베딩 모델 스펙 정의
# ═══════════════════════════════════════════════════════════════════════════

EMBEDDING_MODEL_SPECS = {
    # 한국어 전용 (권장)
    "jhgan/ko-sroberta-multitask": {
        "name": "ko-sroberta",
        "dim": 768,
        "memory_mb": 440,
        "lang": "ko",
        "desc": "한국어 특화, 경량",
    },
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS": {
        "name": "ko-sbert",
        "dim": 768,
        "memory_mb": 440,
        "lang": "ko",
        "desc": "한국어 SBERT",
    },
    "BM-K/KoSimCSE-roberta": {
        "name": "ko-simcse",
        "dim": 768,
        "memory_mb": 440,
        "lang": "ko",
        "desc": "한국어 SimCSE",
    },
    
    # 다국어 (권장)
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "name": "multilingual-minilm",
        "dim": 384,
        "memory_mb": 470,
        "lang": "multi",
        "desc": "다국어, 초경량",
    },
    "intfloat/multilingual-e5-large": {
        "name": "multilingual-e5",
        "dim": 1024,
        "memory_mb": 1200,
        "lang": "multi",
        "desc": "다국어, 고성능",
    },
    "BAAI/bge-m3": {
        "name": "bge-m3",
        "dim": 1024,
        "memory_mb": 1300,
        "lang": "multi",
        "desc": "다국어, 고성능",
    },
    
    # 영어 전용
    "sentence-transformers/all-MiniLM-L6-v2": {
        "name": "minilm",
        "dim": 384,
        "memory_mb": 90,
        "lang": "en",
        "desc": "영어, 초경량",
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "name": "mpnet",
        "dim": 768,
        "memory_mb": 420,
        "lang": "en",
        "desc": "영어, 고품질",
    },
    
    # Qwen Embedding
    "Qwen/Qwen3-Embedding-0.6B": {
        "name": "qwen3-0.6b",
        "dim": 1024,
        "memory_mb": 600,
        "lang": "multi",
        "desc": "Qwen 임베딩, 경량",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# 검색 결과 데이터 클래스
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    """검색 결과 단일 항목"""
    text: str
    similarity: float
    metadata: Dict
    id: str
    confidence: str  # "high", "medium", "low"
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "similarity": self.similarity,
            "metadata": self.metadata,
            "id": self.id,
            "confidence": self.confidence,
        }


@dataclass  
class SearchResponse:
    """검색 응답 전체"""
    results: List[SearchResult]
    query: str
    total_found: int
    filtered_count: int
    quality_summary: Dict
    
    def to_dict(self) -> Dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "query": self.query,
            "total_found": self.total_found,
            "filtered_count": self.filtered_count,
            "quality_summary": self.quality_summary,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 모델 호환성 검사
# ═══════════════════════════════════════════════════════════════════════════

def get_model_spec(model_name: str) -> Optional[Dict]:
    """모델 스펙 조회"""
    return EMBEDDING_MODEL_SPECS.get(model_name)


def is_model_compatible(
    model_name: str, 
    max_dim: int = MAX_EMBEDDING_DIM, 
    max_mem: int = MAX_MEMORY_MB
) -> Tuple[bool, str]:
    """모델 호환성 검사"""
    spec = get_model_spec(model_name)
    
    if spec is None:
        return True, f"⚠️ 알 수 없는 모델: {model_name}. 스펙 확인 필요."
    
    issues = []
    if spec['dim'] > max_dim:
        issues.append(f"dim={spec['dim']} > {max_dim}")
    if spec['memory_mb'] > max_mem:
        issues.append(f"memory={spec['memory_mb']}MB > {max_mem}MB")
    
    if issues:
        return False, f"❌ {model_name} 호환 불가: {', '.join(issues)}"
    
    return True, f"✅ {model_name} 호환 가능 (dim={spec['dim']}, mem={spec['memory_mb']}MB)"


def filter_compatible_models(
    max_dim: int = MAX_EMBEDDING_DIM, 
    max_mem: int = MAX_MEMORY_MB
) -> List[Dict]:
    """호환 가능한 모델 목록"""
    compatible = []
    for model_path, spec in EMBEDDING_MODEL_SPECS.items():
        if spec['dim'] <= max_dim and spec['memory_mb'] <= max_mem:
            compatible.append({"path": model_path, **spec})
    compatible.sort(key=lambda x: x['memory_mb'])
    return compatible


def get_embedding_model_info() -> Dict:
    """임베딩 모델 전체 정보"""
    all_models = []
    compatible = []
    incompatible = []
    
    for model_path, spec in EMBEDDING_MODEL_SPECS.items():
        is_compat = spec['dim'] <= MAX_EMBEDDING_DIM and spec['memory_mb'] <= MAX_MEMORY_MB
        model_info = {"path": model_path, **spec, "compatible": is_compat}
        all_models.append(model_info)
        (compatible if is_compat else incompatible).append(model_info)
    
    return {
        "all": all_models,
        "compatible": compatible,
        "incompatible": incompatible,
        "filter_criteria": {"max_dim": MAX_EMBEDDING_DIM, "max_memory_mb": MAX_MEMORY_MB}
    }


# ═══════════════════════════════════════════════════════════════════════════
# 기본 유틸리티
# ═══════════════════════════════════════════════════════════════════════════

def get_device() -> str:
    """디바이스 확인"""
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def get_client() -> chromadb.PersistentClient:
    """ChromaDB 클라이언트 싱글톤"""
    global _client
    if _client is None:
        Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def get_embedding_model(
    model_name: str = "jhgan/ko-sroberta-multitask",
    check_compatibility: bool = True
):
    """임베딩 모델 로드 (캐싱)"""
    global _embed_models
    
    if model_name in _embed_models:
        return _embed_models[model_name]
    
    # 호환성 검사
    if check_compatibility:
        is_ok, msg = is_model_compatible(model_name)
        if not is_ok:
            raise ValueError(msg)
        print(msg)
    
    print(f"📦 Loading embedding model: {model_name}...")
    device = get_device()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    
    _embed_models[model_name] = (tokenizer, model)
    print(f"✅ Embedding model loaded: {model_name}")
    return tokenizer, model


def embed_text(text: str, model_name: str = "jhgan/ko-sroberta-multitask") -> List[float]:
    """텍스트 임베딩 생성"""
    tokenizer, model = get_embedding_model(model_name)
    device = get_device()
    
    # 텍스트 길이 제한 (토큰화 전)
    MAX_CHARS = 1500
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embedding = (sum_embeddings / sum_mask).cpu().numpy()
    
    return embedding[0].tolist()


def generate_doc_id(text: str, prefix: str = "") -> str:
    """문서 ID 생성"""
    hash_val = hashlib.md5(text.encode()).hexdigest()[:12]
    return f"{prefix}_{hash_val}" if prefix else hash_val


def get_collection_name_for_model(base_name: str, model_name: str) -> str:
    """모델별 컬렉션 이름 생성"""
    model_suffix = model_name.replace("/", "_").replace("-", "_")
    return f"{base_name}__{model_suffix}"


# ═══════════════════════════════════════════════════════════════════════════
# 컬렉션 관리
# ═══════════════════════════════════════════════════════════════════════════

def create_collection(
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = "jhgan/ko-sroberta-multitask"
):
    """컬렉션 생성/가져오기"""
    client = get_client()
    actual_name = get_collection_name_for_model(collection_name, model_name)
    return client.get_or_create_collection(
        name=actual_name,
        metadata={"hnsw:space": "cosine", "embedding_model": model_name}
    )


def list_collections() -> List[str]:
    """모든 컬렉션 목록"""
    client = get_client()
    return [c.name for c in client.list_collections()]


def get_collection_info(collection_name: str) -> Dict:
    """컬렉션 정보"""
    try:
        client = get_client()
        collection = client.get_collection(name=collection_name)
        return {
            "name": collection_name,
            "count": collection.count(),
            "metadata": collection.metadata
        }
    except Exception:
        return {"name": collection_name, "count": 0, "error": "not found"}


def delete_collection(collection_name: str) -> bool:
    """컬렉션 삭제"""
    try:
        client = get_client()
        client.delete_collection(name=collection_name)
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# 문서 저장
# ═══════════════════════════════════════════════════════════════════════════

def add_documents(
    chunks: List[str],
    doc_name: str,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = "jhgan/ko-sroberta-multitask",
    metadata_list: Optional[List[Dict]] = None
) -> Dict:
    """문서 청크들을 ChromaDB에 저장"""
    
    collection = create_collection(collection_name, model_name)
    actual_collection_name = get_collection_name_for_model(collection_name, model_name)
    
    ids, embeddings, metadatas, documents = [], [], [], []
    
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        
        doc_id = generate_doc_id(chunk, f"{doc_name}_{i}")
        embedding = embed_text(chunk, model_name)
        
        # 메타데이터 구성
        meta = metadata_list[i].copy() if metadata_list and i < len(metadata_list) else {}
        
        # SOP 문서 형식이면 지정된 필드 외에는 추가하지 않음
        if meta.get("doc_type") == "SOP":
            pass # 이미 chunker에서 11개 필드를 맞춰줌
        else:
            meta.update({
                "doc_name": doc_name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "model": model_name,
                "char_count": len(chunk),
            })
            
        # None 값 제거 (ChromaDB 호환)
        meta = {k: v for k, v in meta.items() if v is not None}
        
        ids.append(doc_id)
        embeddings.append(embedding)
        documents.append(chunk)
        metadatas.append(meta)
    
    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    return {
        "success": True,
        "doc_name": doc_name,
        "chunks_added": len(ids),
        "collection": actual_collection_name,
        "model": model_name
    }


def add_single_text(
    text: str,
    doc_name: str = "manual_input",
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = "jhgan/ko-sroberta-multitask"
) -> Dict:
    """단일 텍스트 추가"""
    return add_documents([text], doc_name, collection_name, model_name)


# ═══════════════════════════════════════════════════════════════════════════
# 검색 (핵심 개선!)
# ═══════════════════════════════════════════════════════════════════════════

def _classify_confidence(similarity: float) -> str:
    """유사도 기반 신뢰도 분류"""
    if similarity >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    elif similarity >= DEFAULT_SIMILARITY_THRESHOLD:
        return "medium"
    return "low"


def _calculate_quality_summary(results: List[SearchResult]) -> Dict:
    """검색 품질 요약 계산"""
    if not results:
        return {"avg_similarity": 0, "high_count": 0, "medium_count": 0, "low_count": 0}
    
    similarities = [r.similarity for r in results]
    return {
        "avg_similarity": round(sum(similarities) / len(similarities), 4),
        "max_similarity": round(max(similarities), 4),
        "min_similarity": round(min(similarities), 4),
        "high_count": sum(1 for r in results if r.confidence == "high"),
        "medium_count": sum(1 for r in results if r.confidence == "medium"),
        "low_count": sum(1 for r in results if r.confidence == "low"),
    }


def search(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    n_results: int = 5,
    model_name: str = "jhgan/ko-sroberta-multitask",
    filter_doc: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    return_low_confidence: bool = True,
) -> List[Dict]:
    """
    유사 문서 검색 (개선된 버전)
    
    Args:
        query: 검색 쿼리
        collection_name: 컬렉션 이름
        n_results: 반환할 결과 수
        model_name: 임베딩 모델
        filter_doc: 특정 문서만 검색
        similarity_threshold: 최소 유사도 (None이면 기본값 사용)
        return_low_confidence: False면 낮은 신뢰도 결과 제외
    
    Returns:
        검색 결과 리스트 (dict 형태)
    """
    actual_collection_name = get_collection_name_for_model(collection_name, model_name)
    
    try:
        client = get_client()
        collection = client.get_collection(name=actual_collection_name)
    except Exception:
        return []
    
    if collection.count() == 0:
        return []
    
    # 쿼리 임베딩
    query_embedding = embed_text(query, model_name)
    
    # 필터 설정
    where_filter = {"doc_name": filter_doc} if filter_doc else None
    
    # 더 많이 가져와서 필터링 (품질 향상)
    fetch_count = min(n_results * 2, collection.count())
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_count,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    
    # 결과 변환 및 필터링
    threshold = similarity_threshold if similarity_threshold is not None else DEFAULT_SIMILARITY_THRESHOLD
    search_results = []
    
    if results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            distance = results['distances'][0][i] if results['distances'] else 0
            similarity = max(0, min(1, 1 - distance))  # [0, 1] 범위로 클램핑
            
            confidence = _classify_confidence(similarity)
            
            # 낮은 신뢰도 필터링
            if not return_low_confidence and confidence == "low":
                continue
            
            # threshold 미만 필터링 (단, 최소 결과는 보장)
            if similarity < threshold and len(search_results) >= MIN_RESULTS_BEFORE_FILTER:
                continue
            
            search_results.append({
                "text": doc,
                "similarity": round(similarity, 4),
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                "id": results['ids'][0][i] if results['ids'] else None,
                "confidence": confidence,
            })
    
    # 요청한 개수만큼 반환
    return search_results[:n_results]


def search_with_context(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    n_results: int = 3,
    model_name: str = "jhgan/ko-sroberta-multitask",
    filter_doc: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
) -> Tuple[List[Dict], str]:
    """검색 + 컨텍스트 문자열 생성"""
    
    results = search(
        query=query,
        collection_name=collection_name,
        n_results=n_results,
        model_name=model_name,
        filter_doc=filter_doc,
        similarity_threshold=similarity_threshold,
        return_low_confidence=True,  # 컨텍스트 생성시에는 일단 포함
    )
    
    context_parts = []
    for i, r in enumerate(results):
        meta = r.get('metadata', {})
        confidence = r.get('confidence', 'medium')
        
        # 헤더 구성
        header_parts = []
        
        # 문서명
        doc_name = meta.get('doc_name', f'문서 {i+1}')
        header_parts.append(doc_name)
        
        # 조항 정보
        article_num = meta.get('article_num')
        article_type = meta.get('article_type', 'article')
        if article_num:
            if article_type == 'article':
                header_parts.append(f"제{article_num}조")
            elif article_type == 'chapter':
                header_parts.append(f"제{article_num}장")
            else:
                header_parts.append(str(article_num))
        
        # 유사도 및 신뢰도
        sim_str = f"{r['similarity']:.1%}"
        conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
        
        header = f"[{' - '.join(header_parts)}] ({sim_str} {conf_emoji})"
        context_parts.append(f"{header}\n{r['text']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    return results, context


def search_advanced(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    n_results: int = 5,
    model_name: str = "jhgan/ko-sroberta-multitask",
    filter_doc: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
) -> SearchResponse:
    """고급 검색 (품질 메트릭 포함)"""
    
    # 더 많이 가져와서 분석
    all_results = search(
        query=query,
        collection_name=collection_name,
        n_results=n_results * 2,
        model_name=model_name,
        filter_doc=filter_doc,
        similarity_threshold=0.0,  # 일단 다 가져옴
        return_low_confidence=True,
    )
    
    # SearchResult 객체로 변환
    result_objects = [
        SearchResult(
            text=r['text'],
            similarity=r['similarity'],
            metadata=r['metadata'],
            id=r['id'],
            confidence=r['confidence']
        )
        for r in all_results
    ]
    
    # threshold 적용 필터링
    threshold = similarity_threshold or DEFAULT_SIMILARITY_THRESHOLD
    filtered = [r for r in result_objects if r.similarity >= threshold]
    
    # 최소 결과 보장
    if len(filtered) < MIN_RESULTS_BEFORE_FILTER:
        filtered = result_objects[:MIN_RESULTS_BEFORE_FILTER]
    
    # 요청 개수로 제한
    final_results = filtered[:n_results]
    
    return SearchResponse(
        results=final_results,
        query=query,
        total_found=len(all_results),
        filtered_count=len(filtered),
        quality_summary=_calculate_quality_summary(final_results)
    )


# ═══════════════════════════════════════════════════════════════════════════
# 문서 삭제
# ═══════════════════════════════════════════════════════════════════════════

def delete_by_doc_name(
    doc_name: str,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: Optional[str] = None
) -> Dict:
    """문서 이름으로 삭제"""
    
    if model_name:
        actual_name = get_collection_name_for_model(collection_name, model_name)
        try:
            client = get_client()
            collection = client.get_collection(name=actual_name)
            results = collection.get(where={"doc_name": doc_name}, include=["metadatas"])
            if results['ids']:
                collection.delete(ids=results['ids'])
                return {"success": True, "deleted": len(results['ids']), "collection": actual_name}
        except Exception:
            pass
        return {"success": False, "message": "문서를 찾을 수 없음"}
    
    # 모든 관련 컬렉션에서 삭제
    deleted_total = 0
    for col_name in list_collections():
        if col_name.startswith(collection_name + "__"):
            try:
                client = get_client()
                collection = client.get_collection(name=col_name)
                results = collection.get(where={"doc_name": doc_name}, include=["metadatas"])
                if results['ids']:
                    collection.delete(ids=results['ids'])
                    deleted_total += len(results['ids'])
            except Exception:
                continue
    
    if deleted_total > 0:
        return {"success": True, "deleted": deleted_total}
    return {"success": False, "message": "문서를 찾을 수 없음"}


def delete_all(
    collection_name: str = DEFAULT_COLLECTION,
    model_name: Optional[str] = None
) -> Dict:
    """컬렉션 내 모든 문서 삭제"""
    try:
        if model_name:
            actual_name = get_collection_name_for_model(collection_name, model_name)
            delete_collection(actual_name)
            return {"success": True, "message": f"{actual_name} 삭제됨"}
        
        deleted = []
        for col_name in list_collections():
            if col_name.startswith(collection_name + "__") or col_name == collection_name:
                delete_collection(col_name)
                deleted.append(col_name)
        
        return {"success": True, "deleted_collections": deleted}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# 문서 목록
# ═══════════════════════════════════════════════════════════════════════════

def list_documents(
    collection_name: str = DEFAULT_COLLECTION,
    model_name: Optional[str] = None
) -> List[Dict]:
    """저장된 문서 목록"""
    
    docs = {}
    
    if model_name:
        target_collections = [get_collection_name_for_model(collection_name, model_name)]
    else:
        target_collections = [
            col for col in list_collections()
            if col.startswith(collection_name + "__") or col == collection_name
        ]
    
    for col_name in target_collections:
        try:
            client = get_client()
            collection = client.get_collection(name=col_name)
            results = collection.get(include=["metadatas"])
            
            for meta in (results['metadatas'] or []):
                doc_name = meta.get('doc_name', 'unknown')
                model = meta.get('model', 'unknown')
                key = f"{doc_name}|{model}"
                
                if key not in docs:
                    docs[key] = {
                        "doc_name": doc_name,
                        "doc_title": meta.get('doc_title'),
                        "model": model,
                        "collection": col_name,
                        "chunk_count": 0,
                        "chunk_method": meta.get('chunk_method'),
                    }
                docs[key]["chunk_count"] += 1
        except Exception:
            continue
    
    return list(docs.values())
