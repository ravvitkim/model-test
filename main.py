"""
텍스트 유사도 + RAG API v6.0
- Docling 기반 문서 파싱 (표 지원)
- 에러 수정 (similarity_threshold)
- 가독성 개선 메타데이터 (제N조 형식)
- 에이전트 지원
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
import time

# RAG 모듈
from rag import (
    load_document,
    get_supported_extensions,
    create_chunks,
    create_chunks_from_blocks,
    get_available_methods,
    CHUNK_METHODS,
)
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

# 에이전트 모듈
from agent import RAGAgent, create_rag_agent, AgentResponse


app = FastAPI(title="RAG API", version="6.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50
DEFAULT_CHUNK_METHOD = "article"
DEFAULT_N_RESULTS = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.35

PRESET_MODELS = {
    "ko-sroberta": "jhgan/ko-sroberta-multitask",
    "ko-sbert": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    "ko-simcse": "BM-K/KoSimCSE-roberta",
    "multilingual-minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
    "multilingual-e5-small": "intfloat/multilingual-e5-small",
    "bge-m3": "BAAI/bge-m3",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
}

device = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic 모델
# ═══════════════════════════════════════════════════════════════════════════

class SearchRequest(BaseModel):
    query: str
    collection: str = "documents"
    n_results: int = DEFAULT_N_RESULTS
    model: str = "multilingual-e5-small"
    filter_doc: Optional[str] = None
    similarity_threshold: Optional[float] = None


class AskRequest(BaseModel):
    query: str
    collection: str = "documents"
    n_results: int = DEFAULT_N_RESULTS
    embedding_model: str = "multilingual-e5-small"
    llm_model: str = "qwen2.5:3b"
    llm_backend: str = "ollama"
    check_clarification: bool = True
    filter_doc: Optional[str] = None
    similarity_threshold: Optional[float] = None


class AgentRequest(BaseModel):
    query: str
    collection: str = "documents"
    n_results: int = DEFAULT_N_RESULTS
    embedding_model: str = "multilingual-e5-small"
    llm_model: str = "qwen2.5:3b"
    llm_backend: str = "ollama"
    agent_type: str = "basic"  # basic, react, plan_execute
    enable_clarification: bool = True
    filter_doc: Optional[str] = None


class AskChunkRequest(BaseModel):
    query: str
    chunk_text: str
    llm_model: str = "qwen2.5:3b"
    llm_backend: str = "ollama"


class DeleteDocRequest(BaseModel):
    doc_name: str
    collection: str = "documents"


# ═══════════════════════════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════════════════════════

def resolve_model_path(model_key: str) -> str:
    """프리셋 키 → 실제 경로"""
    return PRESET_MODELS.get(model_key, model_key)


def format_metadata_display(metadata: Dict) -> Dict:
    """메타데이터 표시 형식 개선"""
    display = {}

    # 문서명
    if metadata.get('doc_name'):
        display['doc_name'] = metadata['doc_name']

    # 제목
    if metadata.get('doc_title'):
        display['doc_title'] = metadata['doc_title']

    # SOP ID
    if metadata.get('sop_id'):
        display['sop_id'] = metadata['sop_id']

    # 버전
    if metadata.get('version'):
        display['version'] = f"v{metadata['version']}"

    # 섹션 (제N조 형식) - 이미 포맷팅된 경우
    if metadata.get('section'):
        display['section'] = metadata['section']
    # 아직 포맷팅 안 된 경우
    elif metadata.get('article_num'):
        article_num = metadata['article_num']
        article_type = metadata.get('article_type', 'article')
        if article_type == 'article':
            display['section'] = f"제{article_num}조"
        elif article_type == 'chapter':
            display['section'] = f"제{article_num}장"
        elif article_type == 'section':
            display['section'] = f"제{article_num}절"
        else:
            display['section'] = str(article_num)

    # 제목 (블록)
    if metadata.get('title') and metadata.get('title') != metadata.get('doc_title'):
        display['title'] = metadata['title']

    # 페이지
    if metadata.get('page'):
        display['page'] = f"p.{metadata['page']}"

    return display


def interpret_similarity(score: float) -> str:
    """유사도 해석"""
    if score >= 0.85:
        return "매우 유사"
    elif score >= 0.65:
        return "유사"
    elif score >= 0.50:
        return "관련 있음"
    elif score >= 0.35:
        return "약간 관련"
    return "관련 낮음"


def interpret_confidence(confidence: str) -> str:
    """신뢰도 해석"""
    return {
        "high": "🟢 높음",
        "medium": "🟡 보통",
        "low": "🔴 낮음",
    }.get(confidence, "⚪ 알 수 없음")


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - 기본
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "RAG API v6.0",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "ollama_available": OllamaLLM.is_available(),
        "features": [
            "Docling 기반 문서 파싱 (표 지원)",
            "similarity_threshold 검색 필터링",
            "제N조 형식 메타데이터",
            "에이전트 지원 (basic, react, plan_execute)",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": device}


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - 모델 정보
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/models/embedding")
def get_embedding_models():
    """임베딩 모델 정보"""
    return vector_store.get_embedding_model_info()


@app.get("/models/llm")
def get_llm_models():
    """LLM 모델 정보"""
    ollama_running = OllamaLLM.is_available()
    ollama_models = OllamaLLM.list_models() if ollama_running else []

    return {
        "ollama": {
            "server_running": ollama_running,
            "available_models": ollama_models,
            "models": [
                {**m, "available": m["key"] in ollama_models}
                for m in OLLAMA_MODELS
            ],
        },
        "huggingface": {"models": HUGGINGFACE_MODELS},
    }


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - 문서 업로드
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/rag/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form("documents"),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE),
    chunk_method: str = Form(DEFAULT_CHUNK_METHOD),
    model: str = Form("multilingual-e5-small"),
    overlap: int = Form(DEFAULT_OVERLAP),
    semantic_threshold: float = Form(0.5),
    llm_model: str = Form("qwen2.5:3b"),
    llm_backend: str = Form("ollama"),
):
    """문서 업로드 및 벡터화"""
    filename = file.filename
    content = await file.read()
    model_path = resolve_model_path(model)

    # 지원 형식 체크
    ext = filename.split('.')[-1].lower()
    supported = [e.replace('.', '') for e in get_supported_extensions()]
    if ext not in supported:
        raise HTTPException(400, f"지원하지 않는 형식: .{ext}")

    try:
        # 1. 문서 파싱
        parsed_doc = load_document(filename, content)

        # 2. 청킹
        if chunk_method == "article" and parsed_doc.blocks:
            chunks = create_chunks_from_blocks(
                parsed_doc,
                chunk_size=chunk_size,
                overlap=overlap,
                method="recursive"
            )
        else:
            # LLM 함수 준비
            llm_function = None
            if chunk_method == "llm":
                llm_function = lambda p: get_llm_response(p, llm_model, llm_backend, 500)

            # Semantic용 임베딩 함수
            embed_function = None
            if chunk_method == "semantic":
                embed_function = lambda t: vector_store.embed_text(t, model_path)

            chunks = create_chunks(
                parsed_doc.text,
                chunk_size=chunk_size,
                overlap=overlap,
                method=chunk_method,
                embed_function=embed_function,
                llm_function=llm_function,
                semantic_threshold=semantic_threshold,
            )

        if not chunks:
            raise HTTPException(400, "청크 생성 실패")

        # 3. 메타데이터 구성
        chunk_texts = []
        chunk_metadatas = []

        for chunk in chunks:
            chunk_texts.append(chunk.text)

            meta = {
                "doc_name": filename,
                "doc_title": parsed_doc.metadata.get("title"),
                "sop_id": parsed_doc.metadata.get("sop_id"),
                "version": parsed_doc.metadata.get("version"),
                "chunk_method": chunk_method,
                "chunk_index": chunk.index,
                **chunk.metadata,
            }
            chunk_metadatas.append(meta)

        # 4. 벡터 저장
        result = vector_store.add_documents(
            texts=chunk_texts,
            metadatas=chunk_metadatas,
            collection_name=collection,
            model_name=model_path,
        )

        return {
            "success": True,
            "filename": filename,
            "text_length": len(parsed_doc.text),
            "blocks_parsed": len(parsed_doc.blocks),
            "tables_found": len(parsed_doc.tables),
            "chunks_created": len(chunk_texts),
            "chunk_method": chunk_method,
            "collection": collection,
            "model_used": model_path,
            "document_metadata": parsed_doc.metadata,
        }

    except Exception as e:
        raise HTTPException(500, f"업로드 실패: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - 검색
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/rag/search")
def search_documents(request: SearchRequest):
    """유사 문서 검색"""
    model_path = resolve_model_path(request.model)

    results = vector_store.search(
        query=request.query,
        collection_name=request.collection,
        n_results=request.n_results,
        model_name=model_path,
        filter_doc=request.filter_doc,
        similarity_threshold=request.similarity_threshold,
    )

    # 결과 포맷팅
    formatted_results = []
    for r in results:
        formatted_results.append({
            "text": r["text"],
            "similarity": r["similarity"],
            "interpretation": interpret_similarity(r["similarity"]),
            "confidence": r.get("confidence", "medium"),
            "confidence_text": interpret_confidence(r.get("confidence", "medium")),
            "metadata": r["metadata"],
            "metadata_display": format_metadata_display(r["metadata"]),
        })

    # 품질 요약
    quality_summary = {"message": "결과 없음"}
    if formatted_results:
        sims = [r["similarity"] for r in formatted_results]
        quality_summary = {
            "avg_similarity": round(sum(sims) / len(sims), 4),
            "max_similarity": round(max(sims), 4),
            "min_similarity": round(min(sims), 4),
            "high_confidence_count": sum(1 for r in formatted_results if r["confidence"] == "high"),
            "threshold_used": request.similarity_threshold or DEFAULT_SIMILARITY_THRESHOLD,
        }

    return {
        "query": request.query,
        "results": formatted_results,
        "count": len(formatted_results),
        "model_used": model_path,
        "quality_summary": quality_summary,
    }


@app.post("/rag/search/advanced")
def search_advanced(request: SearchRequest):
    """고급 검색"""
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


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - RAG 답변
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/rag/ask")
def ask_with_rag(request: AskRequest):
    """RAG 기반 답변 생성"""
    model_path = resolve_model_path(request.embedding_model)

    # 1. 검색
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
            "answer": "관련 문서를 찾을 수 없습니다.",
            "sources": [],
            "needs_clarification": False,
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
                "sources": [
                    {**r, "metadata_display": format_metadata_display(r.get("metadata", {}))}
                    for r in results
                ],
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
        answer = f"답변 생성 오류: {str(e)}"

    return {
        "query": request.query,
        "answer": answer,
        "needs_clarification": False,
        "sources": [
            {**r, "metadata_display": format_metadata_display(r.get("metadata", {}))}
            for r in results
        ],
        "embedding_model": model_path,
        "llm_model": request.llm_model,
    }


@app.post("/rag/ask-chunk")
def ask_with_chunk(request: AskChunkRequest):
    """단일 청크 기반 답변"""
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
    }


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - 에이전트
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/rag/agent")
def run_agent(request: AgentRequest):
    """에이전트 기반 RAG"""
    model_path = resolve_model_path(request.embedding_model)

    # 검색 함수
    def search_fn(query: str, n: int, filter_doc: Optional[str]):
        return vector_store.search_with_context(
            query=query,
            collection_name=request.collection,
            n_results=n,
            model_name=model_path,
            filter_doc=filter_doc,
        )

    # LLM 함수
    def llm_fn(prompt: str) -> str:
        return get_llm_response(
            prompt=prompt,
            llm_model=request.llm_model,
            llm_backend=request.llm_backend,
            max_tokens=512
        )

    # 되묻기 함수
    def clarify_fn(query: str, options: List[Dict]) -> str:
        return generate_clarification_question(
            query=query,
            options=options,
            llm_model=request.llm_model,
            llm_backend=request.llm_backend
        )

    # 에이전트 생성
    agent = create_rag_agent(
        search_fn=search_fn,
        llm_fn=llm_fn,
        analyze_fn=analyze_search_results,
        clarify_fn=clarify_fn,
        agent_type=request.agent_type,
        enable_clarification=request.enable_clarification,
    )

    # 실행
    try:
        response = agent.run(
            query=request.query,
            n_results=request.n_results,
            filter_doc=request.filter_doc,
        )

        return {
            "query": request.query,
            "answer": response.answer,
            "sources": [
                {**s, "metadata_display": format_metadata_display(s.get("metadata", {}))}
                for s in response.sources
            ],
            "needs_clarification": response.needs_clarification,
            "clarification_options": response.clarification_options,
            "action_taken": response.action_taken,
            "agent_type": request.agent_type,
            "metadata": response.metadata,
        }
    except Exception as e:
        raise HTTPException(500, f"에이전트 실행 오류: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - 문서 관리
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/rag/documents")
def list_documents(collection: str = "documents"):
    """문서 목록"""
    docs = vector_store.list_documents(collection)
    return {"documents": docs, "collection": collection}


@app.delete("/rag/document")
def delete_document(request: DeleteDocRequest):
    """문서 삭제"""
    return vector_store.delete_by_doc_name(
        doc_name=request.doc_name,
        collection_name=request.collection
    )


@app.get("/rag/collections")
def list_collections():
    """컬렉션 목록"""
    collections = vector_store.list_collections()
    return {
        "collections": [
            vector_store.get_collection_info(name)
            for name in collections
        ]
    }


@app.delete("/rag/collection/{collection_name}")
def delete_collection(collection_name: str):
    """컬렉션 삭제"""
    return vector_store.delete_all(collection_name)


@app.get("/rag/supported-formats")
def get_supported_formats():
    """지원 파일 형식"""
    return {"supported_extensions": get_supported_extensions()}


@app.get("/rag/chunk-methods")
def get_chunk_methods():
    """청킹 방법 목록"""
    return {"methods": get_available_methods()}


# ═══════════════════════════════════════════════════════════════════════════
# 서버 실행
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("🖥️  RAG API v6.0")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ CUDA 불가 - CPU 모드")

    if OllamaLLM.is_available():
        models = OllamaLLM.list_models()
        print(f"✅ Ollama: {len(models)}개 모델")
    else:
        print("⚠️ Ollama 미실행")

    model_info = vector_store.get_embedding_model_info()
    print(f"📊 임베딩 모델: 호환 {len(model_info['compatible'])}개")

    print("=" * 60)
    print("""
    URL: http://localhost:8000
    Docs: http://localhost:8000/docs

    v6.0 주요 기능:
    - Docling 기반 문서 파싱 (표 지원)
    - similarity_threshold 검색 필터링
    - 제N조 형식 메타데이터
    - 에이전트 지원 (/rag/agent)
    """)

    uvicorn.run(app, host="0.0.0.0", port=8000)