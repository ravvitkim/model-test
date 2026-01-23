"""
RAG 챗봇 API v6.2
- section_path 계층 추적 지원
- 챗봇 엔드포인트 추가
- 대화 히스토리 지원
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
import time
import uuid

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


app = FastAPI(title="RAG Chatbot API", version="6.2.0")

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

DEFAULT_CHUNK_SIZE = 200
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

# 대화 히스토리 저장 (메모리)
chat_histories: Dict[str, List[Dict]] = {}


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


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    collection: str = "documents"
    n_results: int = DEFAULT_N_RESULTS
    embedding_model: str = "multilingual-e5-small"
    llm_model: str = "qwen2.5:3b"
    llm_backend: str = "ollama"
    filter_doc: Optional[str] = None
    similarity_threshold: Optional[float] = None
    include_sources: bool = True


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


class DeleteDocRequest(BaseModel):
    doc_name: str
    collection: str = "documents"


# ═══════════════════════════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════════════════════════

def resolve_model_path(model_key: str) -> str:
    return PRESET_MODELS.get(model_key, model_key)


def format_metadata_display(metadata: Dict) -> Dict:
    display = {}
    if metadata.get('doc_name'):
        display['doc_name'] = metadata['doc_name']
    if metadata.get('doc_title'):
        display['doc_title'] = metadata['doc_title']
    if metadata.get('sop_id'):
        display['sop_id'] = metadata['sop_id']
    if metadata.get('version'):
        display['version'] = f"v{metadata['version']}"
    if metadata.get('section_path'):
        display['section_path'] = metadata['section_path']
    if metadata.get('section_path_readable'):
        display['section_path_readable'] = metadata['section_path_readable']
    if metadata.get('section'):
        display['section'] = metadata['section']
    elif metadata.get('article_num'):
        article_num = metadata['article_num']
        article_type = metadata.get('article_type', 'article')
        if article_type == 'article':
            display['section'] = f"제{article_num}조"
        elif article_type == 'chapter':
            display['section'] = f"제{article_num}장"
        else:
            display['section'] = str(article_num)
    if metadata.get('title') and metadata.get('title') != metadata.get('doc_title'):
        display['title'] = metadata['title']
    return display


def build_chat_context(history: List[Dict], max_turns: int = 5) -> str:
    recent = history[-max_turns:] if len(history) > max_turns else history
    context_parts = []
    for turn in recent:
        if turn['role'] == 'user':
            context_parts.append(f"사용자: {turn['content']}")
        else:
            context_parts.append(f"AI: {turn['content'][:200]}...")
    return "\n".join(context_parts)


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - 기본
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "RAG Chatbot API v6.2",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "ollama_available": OllamaLLM.is_available(),
        "features": ["section_path 계층 추적", "챗봇 대화 히스토리", "similarity_threshold 검색 필터링"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": device}


@app.get("/models/embedding")
def get_embedding_models():
    return vector_store.get_embedding_model_info()


@app.get("/models/llm")
def get_llm_models():
    ollama_running = OllamaLLM.is_available()
    ollama_models = OllamaLLM.list_models() if ollama_running else []
    return {
        "ollama": {"server_running": ollama_running, "available_models": ollama_models, "models": [{**m, "available": m["key"] in ollama_models} for m in OLLAMA_MODELS]},
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
):
    start_time = time.time()
    try:
        content = await file.read()
        filename = file.filename
        
        parsed_doc = load_document(filename, content)
        
        if chunk_method == "article" and parsed_doc.blocks:
            chunks = create_chunks_from_blocks(parsed_doc, chunk_size=chunk_size, overlap=overlap, method="recursive")
        else:
            chunks = create_chunks(parsed_doc.text, chunk_size=chunk_size, overlap=overlap, method=chunk_method)
            for chunk in chunks:
                chunk.metadata.update({
                    "doc_name": filename,
                    "doc_title": parsed_doc.metadata.get("title"),
                    "sop_id": parsed_doc.metadata.get("sop_id"),
                    "version": parsed_doc.metadata.get("version"),
                })
        
        model_path = resolve_model_path(model)
        texts = [c.text for c in chunks]
        metadatas = [{**c.metadata, "chunk_method": chunk_method, "model": model} for c in chunks]
        
        vector_store.add_documents(texts=texts, metadatas=metadatas, collection_name=collection, model_name=model_path)
        
        return {
            "success": True,
            "filename": filename,
            "doc_title": parsed_doc.metadata.get("title"),
            "sop_id": parsed_doc.metadata.get("sop_id"),
            "chunks": len(chunks),
            "chunk_method": chunk_method,
            "elapsed_seconds": round(time.time() - start_time, 2),
            "sample_metadata": metadatas[0] if metadatas else {},
        }
    except Exception as e:
        raise HTTPException(500, f"업로드 실패: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - 검색
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/rag/search")
def search_documents(request: SearchRequest):
    model_path = resolve_model_path(request.model)
    results = vector_store.search(
        query=request.query,
        collection_name=request.collection,
        n_results=request.n_results,
        model_name=model_path,
        filter_doc=request.filter_doc,
        similarity_threshold=request.similarity_threshold,
    )
    for r in results:
        r["metadata_display"] = format_metadata_display(r.get("metadata", {}))
    return {"query": request.query, "results": results, "count": len(results)}


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - 챗봇
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/chat")
def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    history = chat_histories[session_id]
    history.append({"role": "user", "content": request.message})
    
    model_path = resolve_model_path(request.embedding_model)
    results, context = vector_store.search_with_context(
        query=request.message,
        collection_name=request.collection,
        n_results=request.n_results,
        model_name=model_path,
        filter_doc=request.filter_doc,
        similarity_threshold=request.similarity_threshold,
    )
    
    if not results:
        answer = "관련 문서를 찾을 수 없습니다. 다른 질문을 해주세요."
        history.append({"role": "assistant", "content": answer})
        return {"session_id": session_id, "message": request.message, "answer": answer, "sources": []}
    
    chat_context = build_chat_context(history[:-1])
    
    prompt = f"""당신은 규정(SOP) 전문가 챗봇입니다. 아래 [참고 문서]와 [대화 기록]을 바탕으로 사용자의 질문에 친근하게 답변하세요.

지침:
- 문서에 없는 내용은 답변에 포함하지 마세요.
- 근거가 되는 조항(예: 5.1.1 항목)이 있다면 반드시 언급하세요.
- 정보를 찾을 수 없다면 '해당 문서에서 정보를 찾을 수 없습니다.'라고 답변하세요.

[대화 기록]
{chat_context if chat_context else "(없음)"}

[참고 문서]
{context}

[사용자 질문]
{request.message}

[챗봇 답변]:"""
    
    try:
        answer = get_llm_response(prompt=prompt, llm_model=request.llm_model, llm_backend=request.llm_backend, max_tokens=512)
    except Exception as e:
        answer = f"답변 생성 오류: {str(e)}"
    
    history.append({"role": "assistant", "content": answer})
    if len(history) > 40:
        chat_histories[session_id] = history[-40:]
    
    response = {"session_id": session_id, "message": request.message, "answer": answer}
    if request.include_sources:
        response["sources"] = [
            {
                "text": r.get("text", "")[:300] + "..." if len(r.get("text", "")) > 300 else r.get("text", ""),
                "similarity": r.get("similarity", 0),
                "metadata": r.get("metadata", {}),
                "metadata_display": format_metadata_display(r.get("metadata", {})),
            }
            for r in results
        ]
    return response


@app.get("/chat/history/{session_id}")
def get_chat_history(session_id: str):
    if session_id not in chat_histories:
        return {"session_id": session_id, "history": []}
    return {"session_id": session_id, "history": chat_histories[session_id]}


@app.delete("/chat/history/{session_id}")
def clear_chat_history(session_id: str):
    if session_id in chat_histories:
        del chat_histories[session_id]
    return {"success": True, "session_id": session_id}


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - RAG 답변
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/rag/ask")
def ask_with_rag(request: AskRequest):
    model_path = resolve_model_path(request.embedding_model)
    results, context = vector_store.search_with_context(
        query=request.query,
        collection_name=request.collection,
        n_results=request.n_results,
        model_name=model_path,
        filter_doc=request.filter_doc,
        similarity_threshold=request.similarity_threshold,
    )
    
    if not results:
        return {"query": request.query, "answer": "관련 문서를 찾을 수 없습니다.", "sources": [], "needs_clarification": False}
    
    if request.check_clarification and not request.filter_doc:
        analysis = analyze_search_results(results)
        if analysis['needs_clarification']:
            clarification_text = generate_clarification_question(query=request.query, options=analysis['options'], llm_model=request.llm_model, llm_backend=request.llm_backend)
            return {
                "query": request.query,
                "answer": clarification_text,
                "needs_clarification": True,
                "clarification_options": analysis['options'],
                "sources": [{**r, "metadata_display": format_metadata_display(r.get("metadata", {}))} for r in results],
            }
    
    prompt = build_rag_prompt(request.query, context, language="ko")
    try:
        answer = get_llm_response(prompt=prompt, llm_model=request.llm_model, llm_backend=request.llm_backend, max_tokens=512)
    except Exception as e:
        answer = f"답변 생성 오류: {str(e)}"
    
    return {
        "query": request.query,
        "answer": answer,
        "needs_clarification": False,
        "sources": [{**r, "metadata_display": format_metadata_display(r.get("metadata", {}))} for r in results],
    }


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - 문서 관리
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/rag/documents")
def list_documents(collection: str = "documents"):
    docs = vector_store.list_documents(collection)
    return {"documents": docs, "collection": collection}


@app.delete("/rag/document")
def delete_document(request: DeleteDocRequest):
    return vector_store.delete_by_doc_name(doc_name=request.doc_name, collection_name=request.collection)


@app.get("/rag/collections")
def list_collections():
    collections = vector_store.list_collections()
    return {"collections": [vector_store.get_collection_info(name) for name in collections]}


@app.delete("/rag/collection/{collection_name}")
def delete_collection(collection_name: str):
    return vector_store.delete_all(collection_name)


@app.get("/rag/supported-formats")
def get_supported_formats():
    return {"supported_extensions": get_supported_extensions()}


@app.get("/rag/chunk-methods")
def get_chunk_methods():
    return {"methods": get_available_methods()}


# ═══════════════════════════════════════════════════════════════════════════
# 서버 실행
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("🤖 RAG Chatbot API v6.2")
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
    print("=" * 60)
    print("URL: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)