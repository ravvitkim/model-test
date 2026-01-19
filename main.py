"""
텍스트 유사도 비교 API - 커스텀 모델 지원 + RAG + Ollama + 에이전트
v5.0 - 확장 청킹 + 임베딩 모델 필터링
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re
import time

# RAG 모듈
from rag.document_loader import load_document, get_supported_extensions
from rag.chunker import (
    create_chunks, 
    get_available_methods,
    CHUNK_METHODS,
    split_semantic,
    split_by_llm,
)
from rag import vector_store
from rag.prompt import build_rag_prompt, build_chunk_prompt
from rag.llm import (
    load_llm, 
    get_llm_response,
    OllamaLLM,
    analyze_search_results,
    generate_clarification_question,
    OLLAMA_MODELS,
    HUGGINGFACE_MODELS
)


app = FastAPI(title="Text Similarity + RAG API", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════
# 프리셋 모델 (빠른 선택용) - 호환 모델만 포함
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
    
    # Qwen Embedding (호환 모델만)
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    # "qwen3-4b" 제거 (dim=2560, mem=4GB 초과)
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
    n_results: int = 5
    model: str = "ko-sroberta"
    filter_doc: Optional[str] = None

class AskRequest(BaseModel):
    """에이전트 패턴 RAG 요청"""
    query: str
    collection: str = "documents"
    n_results: int = 5
    embedding_model: str = "ko-sroberta"
    llm_model: str = "qwen2.5:3b"
    llm_backend: str = "ollama"
    check_clarification: bool = True
    filter_doc: Optional[str] = None

class AskChunkRequest(BaseModel):
    query: str
    chunk_text: str
    llm_model: str = "qwen2.5:3b"
    llm_backend: str = "ollama"

class DeleteDocRequest(BaseModel):
    doc_name: str
    collection: str = "documents"

class EmbeddingFilterRequest(BaseModel):
    """임베딩 모델 필터링 요청"""
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
    if score >= 0.9: return "매우 유사함 (거의 동일)"
    elif score >= 0.7: return "유사함 (같은 주제)"
    elif score >= 0.5: return "어느 정도 관련 있음"
    elif score >= 0.3: return "약간 관련 있음"
    return "관련 없음"


def create_embed_function(model_key: str):
    """청킹용 임베딩 함수 생성"""
    (tokenizer, model), _ = load_model(model_key)
    def embed_fn(text: str) -> np.ndarray:
        return embed_text(text, tokenizer, model)
    return embed_fn


def create_llm_function(llm_model: str, llm_backend: str):
    """청킹용 LLM 함수 생성"""
    def llm_fn(prompt: str) -> str:
        return get_llm_response(
            prompt=prompt,
            llm_model=llm_model,
            llm_backend=llm_backend,
            max_tokens=1024
        )
    return llm_fn


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - 기본
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "Text Similarity + RAG API v5.0 (Extended Chunking + Model Filter)",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "ollama_available": OllamaLLM.is_available(),
        "features": [
            "RecursiveCharacterTextSplitter",
            "SemanticSplitter",
            "LLM-based Parsing",
            "Embedding Model Filtering (dim≤1024, mem≤1300MB)"
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
    """LLM 모델 목록 (프론트엔드용)"""
    ollama_available = OllamaLLM.is_available()
    available_ollama_models = OllamaLLM.list_models() if ollama_available else []
    
    ollama_models_with_status = []
    for m in OLLAMA_MODELS:
        ollama_models_with_status.append({
            **m,
            "available": m["key"] in available_ollama_models or any(m["key"].split(":")[0] in a for a in available_ollama_models)
        })
    
    return {
        "ollama": {
            "server_running": ollama_available,
            "available_models": available_ollama_models,
            "models": ollama_models_with_status
        },
        "huggingface": {
            "models": HUGGINGFACE_MODELS
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
# 임베딩 모델 필터링 API ← NEW
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/models/embedding")
def get_embedding_models():
    """
    임베딩 모델 전체 정보 (호환/비호환 분류)
    - 기본 필터: dim ≤ 1024, memory ≤ 1300MB
    """
    return vector_store.get_embedding_model_info()


@app.post("/models/embedding/filter")
def filter_embedding_models(request: EmbeddingFilterRequest):
    """
    커스텀 조건으로 임베딩 모델 필터링
    """
    compatible = vector_store.filter_compatible_models(
        max_dim=request.max_dim,
        max_mem=request.max_memory_mb
    )
    return {
        "filter_criteria": {
            "max_dim": request.max_dim,
            "max_memory_mb": request.max_memory_mb
        },
        "compatible_models": compatible,
        "count": len(compatible)
    }


@app.get("/models/embedding/{model_key}/check")
def check_model_compatibility(model_key: str):
    """
    특정 모델 호환성 검사
    """
    model_path = resolve_model_path(model_key)
    is_ok, msg = vector_store.is_model_compatible(model_path)
    spec = vector_store.get_model_spec(model_path)
    
    return {
        "model_key": model_key,
        "model_path": model_path,
        "compatible": is_ok,
        "message": msg,
        "spec": spec
    }


# ═══════════════════════════════════════════════════════════════════════════
# 청킹 방식 API ← NEW
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/chunking/methods")
def get_chunking_methods():
    """사용 가능한 청킹 방식 목록"""
    return {
        "methods": CHUNK_METHODS,
        "default": "article",
        "recommended_order": [
            {"method": "recursive", "desc": "랭체인 스타일, 범용적"},
            {"method": "semantic", "desc": "의미 기반, 품질 좋음 (느림)"},
            {"method": "sentence", "desc": "문장 단위, 빠름"},
            {"method": "llm", "desc": "LLM 파싱, 가장 정교함 (가장 느림)"},
        ]
    }


# ═══════════════════════════════════════════════════════════════════════════
# 텍스트 비교 API
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/compare")
def compare_texts(request: CompareRequest):
    """두 텍스트 비교"""
    (tokenizer, model), load_time = load_model(request.model)
    
    start_time = time.time()
    emb1 = embed_text(request.text1, tokenizer, model)
    emb2 = embed_text(request.text2, tokenizer, model)
    similarity = calculate_similarity(emb1, emb2)
    inference_time = time.time() - start_time
    
    return {
        "similarity": round(similarity, 4),
        "interpretation": interpret_similarity(similarity),
        "model_used": resolve_model_path(request.model),
        "load_time": round(load_time, 2),
        "inference_time": round(inference_time, 4)
    }


@app.post("/compare/models")
def compare_with_multiple_models(request: MultiModelCompareRequest):
    """여러 모델로 동시 비교"""
    results = []
    
    for model_key in request.models:
        try:
            (tokenizer, model), load_time = load_model(model_key)
            
            start_time = time.time()
            emb1 = embed_text(request.text1, tokenizer, model)
            emb2 = embed_text(request.text2, tokenizer, model)
            similarity = calculate_similarity(emb1, emb2)
            inference_time = time.time() - start_time
            
            results.append({
                "model_key": model_key,
                "model_path": resolve_model_path(model_key),
                "similarity": round(similarity, 4),
                "interpretation": interpret_similarity(similarity),
                "load_time": round(load_time, 2),
                "inference_time": round(inference_time, 4),
                "success": True,
                "error": None
            })
        except Exception as e:
            results.append({
                "model_key": model_key,
                "model_path": resolve_model_path(model_key),
                "similarity": 0,
                "interpretation": "로드 실패",
                "success": False,
                "error": str(e)
            })
    
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return {"results": results, "text1": request.text1, "text2": request.text2}


@app.post("/compare/matrix")
def compare_matrix(request: MatrixRequest):
    """매트릭스 비교"""
    (tokenizer, model), _ = load_model(request.model)
    
    embeddings = [embed_text(t, tokenizer, model) for t in request.texts]
    
    n = len(request.texts)
    matrix = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            matrix[i][j] = round(calculate_similarity(embeddings[i], embeddings[j]), 4)
    
    return {
        "similarity_matrix": matrix,
        "texts": request.texts,
        "model_used": resolve_model_path(request.model)
    }


# ═══════════════════════════════════════════════════════════════════════════
# RAG 엔드포인트
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/rag/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form(default="documents"),
    chunk_size: int = Form(default=300),
    chunk_method: str = Form(default="article"),
    overlap: int = Form(default=50),
    model: str = Form(default="ko-sroberta"),
    semantic_threshold: float = Form(default=0.5),
    llm_model: str = Form(default="qwen2.5:3b"),
    llm_backend: str = Form(default="ollama"),
):
    """
    문서 업로드 및 임베딩 저장
    
    chunk_method 옵션:
    - sentence: 문장 단위
    - paragraph: 문단 단위
    - article: 조항 단위 (SOP/법률)
    - recursive: RecursiveCharacterTextSplitter (랭체인)
    - semantic: 의미 기반 분할
    - llm: LLM 기반 구조 파싱
    """
    content = await file.read()
    filename = file.filename
    model_path = resolve_model_path(model)
    
    # 모델 호환성 검사
    is_ok, msg = vector_store.is_model_compatible(model_path)
    if not is_ok:
        raise HTTPException(status_code=400, detail=msg)
    
    text = load_document(filename, content)
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="문서에서 텍스트를 추출할 수 없습니다.")
    
    # 청킹 함수 준비
    embed_function = None
    llm_function = None
    
    if chunk_method == "semantic":
        embed_function = create_embed_function(model)
    elif chunk_method == "llm":
        llm_function = create_llm_function(llm_model, llm_backend)
    
    # 청킹 수행
    chunks = create_chunks(
        text, 
        chunk_size=chunk_size, 
        overlap=overlap, 
        method=chunk_method,
        embed_function=embed_function,
        llm_function=llm_function,
        semantic_threshold=semantic_threshold,
    )
    chunk_texts = [c.text for c in chunks]
    
    # 메타데이터 구성
    metadata_list = []
    for c in chunks:
        meta = {
            "doc_name": filename,
            **c.metadata
        }
        metadata_list.append(meta)
    
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
        "text_length": len(text),
        "chunks_created": len(chunk_texts),
        "chunk_method": chunk_method,
        "chunk_size": chunk_size,
        "collection": collection,
        "model_used": model_path
    }


@app.post("/rag/search")
def search_documents(request: SearchRequest):
    """유사 문서 검색"""
    model_path = resolve_model_path(request.model)
    
    results = vector_store.search(
        query=request.query,
        collection_name=request.collection,
        n_results=request.n_results,
        model_name=model_path,
        filter_doc=request.filter_doc
    )
    
    return {
        "query": request.query,
        "results": results,
        "count": len(results),
        "model_used": model_path
    }


@app.post("/rag/ask")
def ask_with_agent(request: AskRequest):
    """에이전트 패턴 RAG"""
    model_path = resolve_model_path(request.embedding_model)
    
    # 1. 벡터 검색 수행
    results, context = vector_store.search_with_context(
        query=request.query,
        collection_name=request.collection,
        n_results=request.n_results,
        model_name=model_path,
        filter_doc=request.filter_doc
    )
    
    if not results:
        return {
            "query": request.query,
            "answer": "관련 규정 내용을 찾을 수 없습니다.",
            "sources": [],
            "needs_clarification": False
        }
    
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
                "sources": results
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
    
    return {
        "query": request.query,
        "answer": answer,
        "needs_clarification": False,
        "sources": results,
        "embedding_model": model_path,
        "llm_model": request.llm_model
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
    
    # Ollama 체크
    if OllamaLLM.is_available():
        models = OllamaLLM.list_models()
        print(f"✅ Ollama 서버 실행 중 ({len(models)}개 모델)")
    else:
        print("⚠️  Ollama 서버 미실행 - HuggingFace LLM 사용")
    
    # 임베딩 모델 필터 정보
    model_info = vector_store.get_embedding_model_info()
    print(f"\n📊 임베딩 모델 필터링 (dim≤1024, mem≤1300MB)")
    print(f"   - 호환: {len(model_info['compatible'])}개")
    print(f"   - 비호환: {len(model_info['incompatible'])}개")
    
    print("=" * 60)
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     텍스트 유사도 + RAG API 서버 v5.0                         ║
    ║     (Extended Chunking + Model Filter)                        ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  URL: http://localhost:8000                                   ║
    ║  Docs: http://localhost:8000/docs                             ║
    ║                                                               ║
    ║  📄 청킹 방식:                                                ║
    ║     - sentence: 문장 단위                                     ║
    ║     - paragraph: 문단 단위                                    ║
    ║     - article: 조항 단위 (SOP/법률)                           ║
    ║     - recursive: RecursiveCharacterTextSplitter               ║
    ║     - semantic: 의미 기반 (임베딩 유사도)                     ║
    ║     - llm: LLM 기반 구조 파싱                                 ║
    ║                                                               ║
    ║  🔍 임베딩 모델 필터링:                                       ║
    ║     GET  /models/embedding         - 전체 모델 정보           ║
    ║     POST /models/embedding/filter  - 커스텀 필터              ║
    ║     GET  /models/embedding/{key}/check - 호환성 검사          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000)