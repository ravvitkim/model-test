"""
ChromaDB 벡터 스토어 - 임베딩 저장 및 검색
- 모델별 컬렉션 자동 분리
- 임베딩 모델 필터링 (dim ≤ 1024, mem ≤ 1300MB) ← NEW
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import hashlib
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig


# 기본 설정
DEFAULT_COLLECTION = "documents"
CHROMA_PATH = "./chroma_db"

# 전역 변수
_client = None
_embed_models = {}
_device = None


# ═══════════════════════════════════════════════════════════════════════════
# 임베딩 모델 스펙 정의 (필터링용) ← NEW
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
    
    # Qwen Embedding (주의: 크기 확인 필요)
    "Qwen/Qwen3-Embedding-0.6B": {
        "name": "qwen3-0.6b",
        "dim": 1024,
        "memory_mb": 600,
        "lang": "multi",
        "desc": "Qwen 임베딩, 경량",
    },
    "Qwen/Qwen3-Embedding-4B": {
        "name": "qwen3-4b",
        "dim": 2560,  # ⚠️ 조건 초과!
        "memory_mb": 4000,  # ⚠️ 조건 초과!
        "lang": "multi",
        "desc": "⚠️ dim/mem 초과",
        "warning": True,
    },
}

# 필터링 기준
MAX_EMBEDDING_DIM = 1024
MAX_MEMORY_MB = 1300


def get_model_spec(model_name: str) -> Optional[Dict]:
    """모델 스펙 조회"""
    return EMBEDDING_MODEL_SPECS.get(model_name)


def is_model_compatible(model_name: str, max_dim: int = MAX_EMBEDDING_DIM, max_mem: int = MAX_MEMORY_MB) -> Tuple[bool, str]:
    """
    모델 호환성 검사 (dim ≤ 1024, mem ≤ 1300MB)
    
    Returns:
        (is_compatible, message)
    """
    spec = get_model_spec(model_name)
    
    if spec is None:
        # 알려지지 않은 모델은 일단 허용 (경고만)
        return True, f"⚠️ 알 수 없는 모델: {model_name}. 스펙 확인 필요."
    
    issues = []
    
    if spec['dim'] > max_dim:
        issues.append(f"dim={spec['dim']} > {max_dim}")
    
    if spec['memory_mb'] > max_mem:
        issues.append(f"memory={spec['memory_mb']}MB > {max_mem}MB")
    
    if issues:
        return False, f"❌ {model_name} 호환 불가: {', '.join(issues)}"
    
    return True, f"✅ {model_name} 호환 가능 (dim={spec['dim']}, mem={spec['memory_mb']}MB)"


def filter_compatible_models(max_dim: int = MAX_EMBEDDING_DIM, max_mem: int = MAX_MEMORY_MB) -> List[Dict]:
    """호환 가능한 모델 목록 필터링"""
    compatible = []
    
    for model_path, spec in EMBEDDING_MODEL_SPECS.items():
        if spec['dim'] <= max_dim and spec['memory_mb'] <= max_mem:
            compatible.append({
                "path": model_path,
                **spec
            })
    
    # 메모리 순 정렬
    compatible.sort(key=lambda x: x['memory_mb'])
    return compatible


def get_embedding_model_info() -> Dict:
    """임베딩 모델 전체 정보 (프론트엔드용)"""
    all_models = []
    compatible_models = []
    incompatible_models = []
    
    for model_path, spec in EMBEDDING_MODEL_SPECS.items():
        model_info = {
            "path": model_path,
            **spec,
            "compatible": spec['dim'] <= MAX_EMBEDDING_DIM and spec['memory_mb'] <= MAX_MEMORY_MB
        }
        all_models.append(model_info)
        
        if model_info['compatible']:
            compatible_models.append(model_info)
        else:
            incompatible_models.append(model_info)
    
    return {
        "all": all_models,
        "compatible": compatible_models,
        "incompatible": incompatible_models,
        "filter_criteria": {
            "max_dim": MAX_EMBEDDING_DIM,
            "max_memory_mb": MAX_MEMORY_MB,
        }
    }


def auto_detect_model_spec(model_name: str) -> Optional[Dict]:
    """
    HuggingFace에서 모델 스펙 자동 감지
    (알려지지 않은 모델용)
    """
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # hidden_size 추출 (임베딩 차원)
        dim = getattr(config, 'hidden_size', None)
        if dim is None:
            dim = getattr(config, 'd_model', None)
        
        # 파라미터 수로 메모리 추정 (대략 4bytes per param)
        num_params = getattr(config, 'num_parameters', None)
        if num_params is None:
            # 모델 로드해서 확인
            try:
                model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                num_params = sum(p.numel() for p in model.parameters())
                del model
                torch.cuda.empty_cache()
            except:
                num_params = None
        
        memory_mb = int(num_params * 4 / 1024 / 1024) if num_params else None
        
        return {
            "name": model_name.split("/")[-1],
            "dim": dim,
            "memory_mb": memory_mb,
            "lang": "unknown",
            "desc": "자동 감지",
            "auto_detected": True,
        }
    except Exception as e:
        print(f"⚠️ 모델 스펙 자동 감지 실패: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 기본 유틸리티
# ═══════════════════════════════════════════════════════════════════════════

def get_device():
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


def get_embedding_model(model_name: str = "jhgan/ko-sroberta-multitask", check_compatibility: bool = True):
    """
    임베딩 모델 로드 (모델별 캐싱)
    
    Args:
        model_name: 모델 경로
        check_compatibility: 호환성 검사 여부
    """
    global _embed_models
    
    # 호환성 검사
    if check_compatibility:
        is_ok, msg = is_model_compatible(model_name)
        print(msg)
        if not is_ok:
            raise ValueError(f"모델 호환성 오류: {msg}")
    
    if model_name in _embed_models:
        return _embed_models[model_name]
    
    print(f"📦 임베딩 모델 로딩: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(get_device())
    model.eval()
    print(f"✅ 모델 로드 완료: {model_name} (device: {get_device()})")
    
    _embed_models[model_name] = (tokenizer, model)
    return tokenizer, model


def get_collection_name_for_model(base_name: str, model_name: str) -> str:
    """모델별 컬렉션 이름 생성"""
    safe_model = model_name.split("/")[-1].replace("-", "_").replace(".", "_")
    return f"{base_name}__{safe_model}"


def embed_text(text: str, model_name: str = "jhgan/ko-sroberta-multitask") -> List[float]:
    """텍스트를 임베딩 벡터로 변환"""
    tokenizer, model = get_embedding_model(model_name)
    
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(get_device())
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean Pooling
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embedding = (sum_embeddings / sum_mask).cpu().numpy()
    
    return embedding[0].tolist()


def embed_texts(texts: List[str], model_name: str = "jhgan/ko-sroberta-multitask") -> List[List[float]]:
    """여러 텍스트를 임베딩"""
    return [embed_text(text, model_name) for text in texts]


def generate_doc_id(text: str, doc_name: str = "") -> str:
    """문서 ID 생성"""
    content = f"{doc_name}:{text[:100]}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════
# 컬렉션 관리
# ═══════════════════════════════════════════════════════════════════════════

def create_collection(name: str, model_name: str = None) -> chromadb.Collection:
    """컬렉션 생성 또는 가져오기"""
    client = get_client()
    
    if model_name:
        name = get_collection_name_for_model(name, model_name)
    
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )


def delete_collection(name: str) -> bool:
    """컬렉션 삭제"""
    try:
        client = get_client()
        client.delete_collection(name=name)
        return True
    except Exception:
        return False


def list_collections() -> List[str]:
    """컬렉션 목록"""
    client = get_client()
    return [col.name for col in client.list_collections()]


def get_collection_info(name: str) -> Dict:
    """컬렉션 정보"""
    try:
        client = get_client()
        collection = client.get_collection(name=name)
        
        model_info = "unknown"
        if "__" in name:
            model_info = name.split("__")[-1]
        
        return {
            "name": name,
            "count": collection.count(),
            "model": model_info,
            "metadata": collection.metadata
        }
    except Exception as e:
        return {"name": name, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# 문서 추가
# ═══════════════════════════════════════════════════════════════════════════

def add_documents(
    chunks: List[str],
    doc_name: str,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = "jhgan/ko-sroberta-multitask",
    metadata_list: Optional[List[Dict]] = None
) -> Dict:
    """문서 청크들을 ChromaDB에 저장 (모델별 컬렉션)"""
    
    actual_collection_name = get_collection_name_for_model(collection_name, model_name)
    collection = create_collection(collection_name, model_name)
    
    ids = []
    embeddings = []
    metadatas = []
    documents = []
    
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
            
        doc_id = generate_doc_id(chunk, f"{doc_name}_{i}")
        embedding = embed_text(chunk, model_name)
        
        # 메타데이터 가져오기 또는 생성
        if metadata_list and i < len(metadata_list):
            meta = metadata_list[i].copy()
        else:
            meta = {
                "doc_name": doc_name,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
        
        # 필수 필드 보장
        meta["doc_name"] = meta.get("doc_name", doc_name)
        meta["model"] = model_name
        
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
# 검색
# ═══════════════════════════════════════════════════════════════════════════

def search(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    n_results: int = 5,
    model_name: str = "jhgan/ko-sroberta-multitask",
    filter_doc: str = None
) -> List[Dict]:
    """유사 문서 검색 (모델별 컬렉션)"""
    
    actual_collection_name = get_collection_name_for_model(collection_name, model_name)
    
    try:
        client = get_client()
        collection = client.get_collection(name=actual_collection_name)
    except Exception:
        return []
    
    if collection.count() == 0:
        return []
    
    query_embedding = embed_text(query, model_name)
    
    # 필터 설정
    where_filter = None
    if filter_doc:
        where_filter = {"doc_name": filter_doc}
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    
    search_results = []
    
    if results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            distance = results['distances'][0][i] if results['distances'] else 0
            similarity = 1 - distance
            
            search_results.append({
                "text": doc,
                "similarity": round(max(0, min(1, similarity)), 4),
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                "id": results['ids'][0][i] if results['ids'] else None
            })
    
    return search_results


def search_with_context(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    n_results: int = 3,
    model_name: str = "jhgan/ko-sroberta-multitask",
    filter_doc: str = None
) -> Tuple[List[Dict], str]:
    """검색 + 컨텍스트 생성"""
    results = search(query, collection_name, n_results, model_name, filter_doc)
    
    context_parts = []
    for i, r in enumerate(results):
        meta = r.get('metadata', {})
        
        # 조항 정보 포함
        header = f"[문서 {i+1}]"
        if meta.get('doc_name'):
            header = f"[{meta['doc_name']}"
            if meta.get('article_num'):
                article_type = meta.get('article_type', 'article')
                if article_type == 'article':
                    header += f" - 제{meta['article_num']}조"
                elif article_type == 'chapter':
                    header += f" - 제{meta['article_num']}장"
                else:
                    header += f" - {meta['article_num']}"
            header += f"] (유사도: {r['similarity']:.1%})"
        
        context_parts.append(f"{header}\n{r['text']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    return results, context


# ═══════════════════════════════════════════════════════════════════════════
# 문서 삭제
# ═══════════════════════════════════════════════════════════════════════════

def delete_by_doc_name(
    doc_name: str, 
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = None
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
        except:
            pass
        return {"success": False, "message": "문서를 찾을 수 없음"}
    
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
            except:
                continue
    
    if deleted_total > 0:
        return {"success": True, "deleted": deleted_total}
    return {"success": False, "message": "문서를 찾을 수 없음"}


def delete_all(collection_name: str = DEFAULT_COLLECTION, model_name: str = None) -> Dict:
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

def list_documents(collection_name: str = DEFAULT_COLLECTION, model_name: str = None) -> List[Dict]:
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
        except:
            continue
    
    return list(docs.values())