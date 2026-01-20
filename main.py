"""
텍스트 유사도 비교 API - 리팩토링 v5.1
- document_loader 직접 사용 (블록 기반 청킹 정상 작동)
- 검색 품질 지표 추가
- 청크 크기 기본값 300 (한국어 최적화)
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import time

# RAG 모듈 (정확한 import)
from rag.document_loader import load_document, get_supported_extensions
from rag.chunker import (
    create_chunks, 
    create_chunks_from_blocks,
    get_available_methods, 
    CHUNK_METHODS,
    Chunk,
)
from rag.parser import ParsedDocument  # 여기서 import (중복 정의 제거!)
from rag import vector_store
from rag.prompt import build_rag_prompt, build_chunk_prompt
from rag.llm import (
    get_llm_response,
    OllamaLLM,
    analyze_search_results,
    generate_clarification_question,
    OLLAMA_MODELS,
    HUGGINGFACE_MODELS,
)


app = FastAPI(title="Text Similarity + RAG API", version="5.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
# 설정 상수
# ═══════════════════════════════════════════════════════════════════════════

# 청킹 기본값 (한국어 최적화)
DEFAULT_CHUNK_SIZE = 300   # 기존 500 → 300 (한국어에서 더 정확)
DEFAULT_OVERLAP = 50
DEFAULT_CHUNK_METHOD = "article"  # SOP 문서용 기본값

# 검색 기본값
DEFAULT_N_RESULTS = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.35


# ═══════════════════════════════════════════════════════════════════════════
# 프리셋 모델
# ═══════════════════════════════════════════════════════════════════════════

PRESET_MODELS = {
    # 한국어 전용 (권장)
    "ko-sroberta": "jhgan/ko-sroberta-multitask",
    "ko-sbert": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    "ko-simcse": "BM-K/KoSimCSE-roberta",
    
    # 다국어
    "multilingual-minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "multilingual-e5": "intfloat/multilingual-e5-large",
    "bge-m3": "BAAI/bge-m3",
    
    # 영어 전용
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    
    # Qwen Embedding
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
}

# 전역 변수
loaded_models = {}
device = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic 모델
# ═══════════════════════════════════════════════════════════════════════════

class CompareRequest(BaseModel):
    text1: str
    text2: str
    model: str = "ko-sroberta"


class MultiModelCompareRequest(BaseModel):
    text1: str
    text2: str
    models: List[str]


class MatrixRequest(BaseModel):
    texts: List[str]
    model: str = "ko-sroberta"


class SearchRequest(BaseModel):
    query: str
    collection: str = "documents"
    n_results: int = DEFAULT_N_RESULTS
    model: str = "ko-sroberta"
    filter_doc: Optional[str] = None
    similarity_threshold: Optional[float] = None  # 추가!


class AskRequest(BaseModel):
    """에이전트 패턴 RAG 요청"""
    query: str
    collection: str = "documents"
    n_results: int = DEFAULT_N_RESULTS
    embedding_model: str = "ko-sroberta"
    llm_model: str = "qwen2.5:3b"
    llm_backend: str = "ollama"
    check_clarification: bool = True
    filter_doc: Optional[str] = None
    similarity_threshold: Optional[float] = None  # 추가!


class AskChunkRequest(BaseModel):
    query: str
    chunk_text: str
    llm_model: str = "qwen2.5:3b"
    llm_backend: str = "ollama"


class DeleteDocRequest(BaseModel):
    doc_name: str
    collection: str = "documents"


class EmbeddingFilterRequest(BaseModel):
    max_dim: int = Field(default=1024, description="최대 임베딩 차원")
    max_memory_mb: int = Field(default=1300, description="최대 메모리 (MB)")


# ═══════════════════════════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════════════════════════

def resolve_model_path(model_key: str) -> str:
    """프리셋 키면 실제 경로로 변환"""
    return PRESET_MODELS.get(model_key, model_key)


def load_model(model_key: str):
    """임베딩 모델 로드"""
    model_path = resolve_model_path(model_key)
    
    if model_path in loaded_models:
        return loaded_models[model_path], 0.0
    
    print(f"📦 Loading embedding model: {model_path}...")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
    model.eval()
    
    load_time = time.time() - start_time
    loaded_models[model_path] = (tokenizer, model)
    print(f"✅ Embedding model loaded: {model_path} ({load_time:.2f}s)")
    
    return (tokenizer, model), load_time


def embed_text(text: str, tokenizer, model) -> np.ndarray:
    """텍스트 임베딩"""
    MAX_TEXT_LENGTH = 1500
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
    
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    embedding = (sum_embeddings / sum_mask).cpu().numpy()
    return embedding[0]


def calculate_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float(cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0])


def interpret_similarity(score: float) -> str:
    """유사도 해석 (개선)"""
    if score >= 0.85:
        return "매우 유사함 (거의 동일)"
    elif score >= 0.65:
        return "유사함 (같은 주제, 높은 관련성)"
    elif score >= 0.50:
        return "관련 있음 (부분적 유사)"
    elif score >= 0.35:
        return "약간 관련 있음"
    return "관련 없음"


def interpret_confidence(confidence: str) -> str:
    """신뢰도 한글 해석"""
    return {
        "high": "🟢 높음 (신뢰할 수 있음)",
        "medium": "🟡 보통 (참고용)",
        "low": "🔴 낮음 (관련성 낮을 수 있음)",
    }.get(confidence, "⚪ 알 수 없음")


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - 기본
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "Text Similarity + RAG API v5.1 (Refactored)",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "ollama_available": OllamaLLM.is_available(),
        "defaults": {
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_method": DEFAULT_CHUNK_METHOD,
            "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
        },
        "improvements": [
            "검색 결과에 신뢰도(confidence) 표시",
            "유사도 threshold 필터링",
            "청크 크기 300 (한국어 최적화)",
            "블록 기반 청킹 정상 작동",
        ]
    }


@app.get("/models")
def get_models():
    """사용 가능한 모델 목록"""
    return {
        "preset_embedding_models": PRESET_MODELS,
        "loaded_embedding_models": list(loaded_models.keys()),
        "ollama": {
            "available": OllamaLLM.is_available(),
            "models": OllamaLLM.list_models() if OllamaLLM.is_available() else [],
            "preset_models": OLLAMA_MODELS
        },
        "huggingface_llm_models": HUGGINGFACE_MODELS,
        "device": device
    }


@app.get("/models/llm")
def get_llm_models():
    """LLM 모델 목록"""
    ollama_available = OllamaLLM.is_available()
    available_ollama_models = OllamaLLM.list_models() if ollama_available else []
    
    ollama_models_with_status = []
    for m in OLLAMA_MODELS:
        ollama_models_with_status.append({
            **m,
            "installed": m["key"] in available_ollama_models
        })
    
    return {
        "ollama": {
            "available": ollama_available,
            "models": ollama_models_with_status,
        },
        "huggingface": HUGGINGFACE_MODELS,
    }


@app.get("/models/embedding")
def get_embedding_models():
    """임베딩 모델 정보"""
    return vector_store.get_embedding_model_info()


@app.post("/models/embedding/filter")
def filter_embedding_models(request: EmbeddingFilterRequest):
    """호환 가능한 임베딩 모델 필터링"""
    compatible = vector_store.filter_compatible_models(
        max_dim=request.max_dim,
        max_mem=request.max_memory_mb
    )
    return {
        "compatible_models": compatible,
        "filter_criteria": {
            "max_dim": request.max_dim,
            "max_memory_mb": request.max_memory_mb
        }
    }


@app.get("/models/embedding/{model_key}/check")
def check_embedding_model(model_key: str):
    """특정 임베딩 모델 호환성 검사"""
    model_path = resolve_model_path(model_key)
    is_ok, message = vector_store.is_model_compatible(model_path)
    return {
        "model_key": model_key,
        "model_path": model_path,
        "compatible": is_ok,
        "message": message
    }


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - 텍스트 유사도
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/compare")
def compare_texts(request: CompareRequest):
    """두 텍스트 유사도 비교"""
    model_path = resolve_model_path(request.model)
    (tokenizer, model), load_time = load_model(model_path)
    
    start_time = time.time()
    emb1 = embed_text(request.text1, tokenizer, model)
    emb2 = embed_text(request.text2, tokenizer, model)
    similarity = calculate_similarity(emb1, emb2)
    inference_time = time.time() - start_time
    
    return {
        "similarity": round(similarity, 4),
        "interpretation": interpret_similarity(similarity),
        "model_used": model_path,
        "load_time": round(load_time, 2),
        "inference_time": round(inference_time, 3),
    }


@app.post("/compare/multi")
def compare_multi_model(request: MultiModelCompareRequest):
    """여러 모델로 유사도 비교"""
    results = []
    
    for model_key in request.models:
        try:
            model_path = resolve_model_path(model_key)
            (tokenizer, model), _ = load_model(model_path)
            
            emb1 = embed_text(request.text1, tokenizer, model)
            emb2 = embed_text(request.text2, tokenizer, model)
            similarity = calculate_similarity(emb1, emb2)
            
            results.append({
                "model": model_key,
                "model_path": model_path,
                "similarity": round(similarity, 4),
                "interpretation": interpret_similarity(similarity),
            })
        except Exception as e:
            results.append({
                "model": model_key,
                "error": str(e)
            })
    
    return {"results": results}


@app.post("/matrix")
def similarity_matrix(request: MatrixRequest):
    """텍스트 간 유사도 행렬"""
    model_path = resolve_model_path(request.model)
    (tokenizer, model), _ = load_model(model_path)
    
    embeddings = [embed_text(t, tokenizer, model) for t in request.texts]
    emb_array = np.array(embeddings)
    
    matrix = cosine_similarity(emb_array)
    
    return {
        "matrix": matrix.tolist(),
        "texts": request.texts,
        "model_used": model_path
    }


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - RAG
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/rag/chunk-methods")
def get_chunk_methods():
    """청킹 방법 목록"""
    return {
        "methods": CHUNK_METHODS,
        "default": DEFAULT_CHUNK_METHOD,
        "default_chunk_size": DEFAULT_CHUNK_SIZE,
        "recommended_for_korean_sop": "article",
    }


@app.post("/rag/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form("documents"),
    model: str = Form("ko-sroberta"),
    chunk_method: str = Form(DEFAULT_CHUNK_METHOD),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE),
    overlap: int = Form(DEFAULT_OVERLAP),
):
    """
    문서 업로드 및 벡터 저장 (수정됨!)
    
    핵심 변경: document_loader.load_document() 직접 사용
    """
    model_path = resolve_model_path(model)
    
    # 파일 확장자 확인
    filename = file.filename
    supported = get_supported_extensions()
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in supported:
        raise HTTPException(400, f"지원하지 않는 파일 형식: {ext}. 지원: {supported}")
    
    content = await file.read()
    
    # 1️⃣ document_loader.load_document() 사용 (핵심 수정!)
    # 이제 ParsedDocument가 blocks를 제대로 포함
    parsed_doc = load_document(filename, content)
    
    print(f"📄 파싱 완료: {filename}")
    print(f"   - 전체 텍스트 길이: {len(parsed_doc.text)}")
    print(f"   - 블록 수: {len(parsed_doc.blocks)}")
    
    # 2️⃣ 블록 기반 청킹 (블록이 있는 경우) 또는 일반 청킹
    if parsed_doc.blocks:
        # 블록 기반 청킹 (메타데이터 보존)
        chunks = create_chunks_from_blocks(
            parsed_doc,
            chunk_size=chunk_size,
            overlap=overlap,
            method="recursive" if chunk_method != "article" else "recursive"
        )
        print(f"   - 블록 기반 청킹: {len(chunks)}개 청크")
    else:
        # 일반 청킹 (블록이 없는 경우)
        chunks = create_chunks(
            parsed_doc.text,
            chunk_size=chunk_size,
            overlap=overlap,
            method=chunk_method
        )
        print(f"   - 일반 청킹: {len(chunks)}개 청크")
    
    # 3️⃣ 메타데이터 구성
    chunk_texts = []
    metadata_list = []
    
    for c in chunks:
        chunk_texts.append(c.text)
        
        # Chunk 객체의 메타데이터 + 문서 메타데이터 병합
        meta = {
            "doc_name": filename,
            "doc_title": parsed_doc.metadata.get("title", filename),
            "chunk_method": chunk_method,
            **c.metadata  # Chunk의 메타데이터 (article_num, article_type 등)
        }
        metadata_list.append(meta)
    
    # 4️⃣ 벡터 저장
    result = vector_store.add_documents(
        chunks=chunk_texts,
        doc_name=filename,
        collection_name=collection,
        model_name=model_path,
        metadata_list=metadata_list
    )
    
    return {
        "success": True,
        "filename": filename,
        "text_length": len(parsed_doc.text),
        "blocks_parsed": len(parsed_doc.blocks),
        "chunks_created": len(chunk_texts),
        "chunk_method": chunk_method,
        "chunk_size": chunk_size,
        "collection": collection,
        "model_used": model_path,
        "document_metadata": parsed_doc.metadata,
    }


@app.post("/rag/search")
def search_documents(request: SearchRequest):
    """유사 문서 검색 (개선됨!)"""
    model_path = resolve_model_path(request.model)
    
    results = vector_store.search(
        query=request.query,
        collection_name=request.collection,
        n_results=request.n_results,
        model_name=model_path,
        filter_doc=request.filter_doc,
        similarity_threshold=request.similarity_threshold,
    )
    
    # 결과에 해석 추가
    for r in results:
        r["interpretation"] = interpret_similarity(r["similarity"])
        r["confidence_text"] = interpret_confidence(r.get("confidence", "medium"))
    
    # 품질 요약
    if results:
        similarities = [r["similarity"] for r in results]
        quality_summary = {
            "avg_similarity": round(sum(similarities) / len(similarities), 4),
            "max_similarity": round(max(similarities), 4),
            "min_similarity": round(min(similarities), 4),
            "high_confidence_count": sum(1 for r in results if r.get("confidence") == "high"),
            "threshold_used": request.similarity_threshold or DEFAULT_SIMILARITY_THRESHOLD,
        }
    else:
        quality_summary = {"message": "결과 없음"}
    
    return {
        "query": request.query,
        "results": results,
        "count": len(results),
        "model_used": model_path,
        "quality_summary": quality_summary,
    }


@app.post("/rag/search/advanced")
def search_advanced(request: SearchRequest):
    """고급 검색 (품질 메트릭 상세)"""
    model_path = resolve_model_path(request.model)
    
    response = vector_store.search_advanced(
        query=request.query,
        collection_name=request.collection,
        n_results=request.n_results,
        model_name=model_path,
        filter_doc=request.filter_doc,
        similarity_threshold=request.similarity_threshold,
    )
    
    return response.to_dict()


@app.post("/rag/ask")
def ask_with_agent(request: AskRequest):
    """에이전트 패턴 RAG (개선됨!)"""
    model_path = resolve_model_path(request.embedding_model)
    
    # 1. 벡터 검색 (threshold 적용)
    results, context = vector_store.search_with_context(
        query=request.query,
        collection_name=request.collection,
        n_results=request.n_results,
        model_name=model_path,
        filter_doc=request.filter_doc,
        similarity_threshold=request.similarity_threshold,
    )
    
    if not results:
        return {
            "query": request.query,
            "answer": "관련 규정 내용을 찾을 수 없습니다.",
            "sources": [],
            "needs_clarification": False,
            "quality": {"message": "검색 결과 없음"},
        }
    
    # 품질 체크: 모든 결과가 low confidence면 경고
    all_low = all(r.get("confidence") == "low" for r in results)
    quality_warning = None
    if all_low:
        quality_warning = "⚠️ 검색된 모든 결과의 관련성이 낮습니다. 질문을 더 구체적으로 해주세요."
    
    # 2. 되묻기 분석
    if request.check_clarification and not request.filter_doc:
        analysis = analyze_search_results(results)
        
        if analysis['needs_clarification']:
            clarification_text = generate_clarification_question(
                query=request.query,
                options=analysis['options'],
                llm_model=request.llm_model,
                llm_backend=request.llm_backend
            )
            
            return {
                "query": request.query,
                "answer": clarification_text,
                "needs_clarification": True,
                "clarification_options": analysis['options'],
                "sources": results,
                "quality_warning": quality_warning,
            }
    
    # 3. 답변 생성
    prompt = build_rag_prompt(request.query, context, language="ko")
    
    try:
        answer = get_llm_response(
            prompt=prompt,
            llm_model=request.llm_model,
            llm_backend=request.llm_backend,
            max_tokens=512
        )
    except Exception as e:
        answer = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    # 결과에 해석 추가
    for r in results:
        r["confidence_text"] = interpret_confidence(r.get("confidence", "medium"))
    
    return {
        "query": request.query,
        "answer": answer,
        "needs_clarification": False,
        "sources": results,
        "embedding_model": model_path,
        "llm_model": request.llm_model,
        "quality_warning": quality_warning,
        "quality": {
            "high_confidence_sources": sum(1 for r in results if r.get("confidence") == "high"),
            "total_sources": len(results),
        }
    }


@app.post("/rag/ask-llm")
def ask_llm_simple(request: AskRequest):
    """단순 RAG (되묻기 없이)"""
    request.check_clarification = False
    return ask_with_agent(request)


@app.post("/rag/ask-chunk")
def ask_with_single_chunk(request: AskChunkRequest):
    """개별 청크 기반 LLM 답변"""
    prompt = build_chunk_prompt(request.query, request.chunk_text, language="ko")
    
    try:
        answer = get_llm_response(
            prompt=prompt,
            llm_model=request.llm_model,
            llm_backend=request.llm_backend,
            max_tokens=200
        )
    except Exception as e:
        answer = f"오류: {str(e)}"
    
    return {
        "query": request.query,
        "answer": answer,
        "llm_model": request.llm_model,
        "llm_backend": request.llm_backend
    }


@app.get("/rag/documents")
def list_documents(collection: str = "documents"):
    """저장된 문서 목록"""
    docs = vector_store.list_documents(collection)
    return {"documents": docs, "collection": collection}


@app.delete("/rag/document")
def delete_document(request: DeleteDocRequest):
    """문서 삭제"""
    result = vector_store.delete_by_doc_name(
        doc_name=request.doc_name,
        collection_name=request.collection
    )
    return result


@app.get("/rag/collections")
def list_collections():
    """컬렉션 목록"""
    collections = vector_store.list_collections()
    collection_info = [vector_store.get_collection_info(name) for name in collections]
    return {"collections": collection_info}


@app.delete("/rag/collection/{collection_name}")
def delete_collection(collection_name: str):
    """컬렉션 삭제"""
    result = vector_store.delete_all(collection_name)
    return result


@app.get("/rag/supported-formats")
def get_supported_formats():
    """지원 파일 형식"""
    return {"supported_extensions": get_supported_extensions()}


@app.delete("/models/cache")
def clear_model_cache():
    """모델 캐시 클리어"""
    global loaded_models
    count = len(loaded_models)
    loaded_models = {}
    torch.cuda.empty_cache()
    return {"message": f"Cleared {count} models from cache"}


# ═══════════════════════════════════════════════════════════════════════════
# 서버 실행
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("🖥️  시스템 정보")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA 사용 가능")
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        print(f"   - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ CUDA 사용 불가 - CPU 모드")
    
    if OllamaLLM.is_available():
        models = OllamaLLM.list_models()
        print(f"✅ Ollama 서버 실행 중 ({len(models)}개 모델)")
    else:
        print("⚠️  Ollama 서버 미실행 - HuggingFace LLM 사용")
    
    model_info = vector_store.get_embedding_model_info()
    print(f"\n📊 임베딩 모델 필터링 (dim≤1024, mem≤1300MB)")
    print(f"   - 호환: {len(model_info['compatible'])}개")
    print(f"   - 비호환: {len(model_info['incompatible'])}개")
    
    print("=" * 60)
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     텍스트 유사도 + RAG API 서버 v5.1 (Refactored)            ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  URL: http://localhost:8000                                   ║
    ║  Docs: http://localhost:8000/docs                             ║
    ║                                                               ║
    ║  📝 v5.1 개선사항:                                            ║
    ║     - 검색 결과 신뢰도(confidence) 표시                       ║
    ║     - 유사도 threshold 필터링 (기본 0.35)                     ║
    ║     - 청크 크기 300 (한국어 최적화)                           ║
    ║     - 블록 기반 청킹 정상 작동                                ║
    ║                                                               ║
    ║  🔍 왜 5개 중 일부만 괜찮은 결과?                             ║
    ║     → confidence: high/medium/low 확인하세요!                 ║
    ║     → similarity_threshold 파라미터로 필터링 가능             ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
