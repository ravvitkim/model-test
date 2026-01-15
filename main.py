"""
텍스트 유사도 비교 API - 커스텀 모델 지원 + RAG
[원문] → [파싱: 품사 분석] → [청킹: 의미 단위] → [임베딩: 벡터 변환] → [코사인 유사도]
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import re
import time

# RAG 모듈
from rag.document_loader import load_document, get_supported_extensions
from rag.chunker import create_chunks
from rag import vector_store

# LLM + Prompt
from rag.llm import load_llm
from rag.prompt import build_rag_prompt, build_chunk_prompt


app = FastAPI(title="Text Similarity API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════
# 프리셋 모델 (빠른 선택용)
# ═══════════════════════════════════════════════════════════════════════════

PRESET_MODELS = {
    # 한국어 전용
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
    "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
}

# 전역 변수
loaded_models = {}
loaded_llms = {}
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

class AddModelRequest(BaseModel):
    key: str
    model_path: str

class ProcessedResult(BaseModel):
    original: str
    pos_tags: List[List[str]]
    chunks: List[str]

class CompareResponse(BaseModel):
    similarity: float
    interpretation: str
    text1_processed: ProcessedResult
    text2_processed: ProcessedResult
    model_used: str
    load_time: float
    inference_time: float

class MultiModelResponse(BaseModel):
    results: List[Dict]
    text1: str
    text2: str

class MatrixResponse(BaseModel):
    similarity_matrix: List[List[float]]
    texts: List[str]
    model_used: str

# ═══════════════════════════════════════════════════════════════════════════
# 모델 로딩 (동적)
# ═══════════════════════════════════════════════════════════════════════════

def resolve_model_path(model_key: str) -> str:
    """프리셋 키면 실제 경로로 변환, 아니면 그대로 반환"""
    return PRESET_MODELS.get(model_key, model_key)

def load_model(model_key: str):
    """임베딩 모델 로드 (캐싱)"""
    model_path = resolve_model_path(model_key)
    
    if model_path in loaded_models:
        return loaded_models[model_path], 0.0
    
    print(f"📦 Loading embedding model: {model_path}...")
    start_time = time.time()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        model.eval()
        
        load_time = time.time() - start_time
        loaded_models[model_path] = (tokenizer, model)
        print(f"✅ Embedding model loaded: {model_path} ({load_time:.2f}s)")
        
        return (tokenizer, model), load_time
    except Exception as e:
        raise ValueError(f"모델 로드 실패: {model_path} - {str(e)}")


def load_llm_model(model_name: str):
    """LLM 모델 로드 (캐싱)"""
    if model_name in loaded_llms:
        return loaded_llms[model_name]
    
    print(f"🤖 Loading LLM: {model_name}...")
    start_time = time.time()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.float16 if device == "cuda" else torch.float32,  # torch_dtype → dtype
            device_map="auto" if device == "cuda" else None
        )
        
        if device == "cpu":
            model = model.to(device)
        
        model.eval()
        
        load_time = time.time() - start_time
        loaded_llms[model_name] = (tokenizer, model)
        print(f"✅ LLM loaded: {model_name} ({load_time:.2f}s)")
        
        return tokenizer, model
    except Exception as e:
        raise ValueError(f"LLM 로드 실패: {model_name} - {str(e)}")

# ═══════════════════════════════════════════════════════════════════════════
# 파이프라인 함수들
# ═══════════════════════════════════════════════════════════════════════════

def parse_pos(text: str, tokenizer) -> List[List[str]]:
    """Stage 1: 품사 분석"""
    tokens = tokenizer.tokenize(text)
    pos_tags = []
    
    for token in tokens:
        clean_token = token.replace("##", "").replace("▁", "").replace("Ġ", "")
        if not clean_token:
            continue
        
        if clean_token.isdigit():
            pos = "NUM"
        elif not clean_token.isalnum():
            pos = "PUNCT"
        elif clean_token.isascii() and clean_token.isalpha():
            pos = "WORD_EN"
        else:
            pos = "WORD_KO"
        
        pos_tags.append([clean_token, pos])
    
    return pos_tags

def chunk_text(text: str) -> List[str]:
    """Stage 2: 의미 단위 청킹"""
    sentences = re.split(r'(?<=[.!?。])\s+', text)
    chunks = []
    
    for sentence in sentences:
        if len(sentence) > 100:
            sub_chunks = re.split(r'[,，]|\s+(그리고|그러나|하지만|또는|및|and|but|or)\s+', sentence)
            chunks.extend([c.strip() for c in sub_chunks if c and len(c) > 2])
        else:
            if sentence.strip():
                chunks.append(sentence.strip())
    
    return chunks

def embed_text(text: str, tokenizer, model) -> np.ndarray:
    """Stage 3: 임베딩 (Mean Pooling)"""
    # 텍스트가 너무 길면 자르기
    MAX_TEXT_LENGTH = 1500  # 약 500~700 토큰
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
    """Stage 4: 코사인 유사도"""
    return float(cosine_similarity(
        emb1.reshape(1, -1),
        emb2.reshape(1, -1)
    )[0][0])

def interpret_similarity(score: float) -> str:
    """유사도 해석"""
    if score >= 0.9:
        return "매우 유사함 (거의 동일)"
    elif score >= 0.7:
        return "유사함 (같은 주제)"
    elif score >= 0.5:
        return "어느 정도 관련 있음"
    elif score >= 0.3:
        return "약간 관련 있음"
    else:
        return "관련 없음"

# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "Text Similarity API v3.0",
        "endpoints": ["/compare", "/compare/models", "/compare/matrix", "/models", "/rag/*"],
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

@app.get("/models")
def get_models():
    """사용 가능한 프리셋 모델 + 로드된 모델"""
    return {
        "preset_models": PRESET_MODELS,
        "loaded_embedding_models": list(loaded_models.keys()),
        "loaded_llm_models": list(loaded_llms.keys()),
        "device": device
    }

@app.post("/models/add")
def add_preset_model(request: AddModelRequest):
    """프리셋에 새 모델 추가"""
    PRESET_MODELS[request.key] = request.model_path
    return {"message": f"Added {request.key}: {request.model_path}", "presets": PRESET_MODELS}

@app.post("/compare", response_model=CompareResponse)
def compare_texts(request: CompareRequest):
    """두 텍스트 비교"""
    try:
        (tokenizer, model), load_time = load_model(request.model)
        
        start_time = time.time()
        
        pos1 = parse_pos(request.text1, tokenizer)
        pos2 = parse_pos(request.text2, tokenizer)
        chunks1 = chunk_text(request.text1)
        chunks2 = chunk_text(request.text2)
        emb1 = embed_text(request.text1, tokenizer, model)
        emb2 = embed_text(request.text2, tokenizer, model)
        similarity = calculate_similarity(emb1, emb2)
        
        inference_time = time.time() - start_time
        
        return CompareResponse(
            similarity=round(similarity, 4),
            interpretation=interpret_similarity(similarity),
            text1_processed=ProcessedResult(
                original=request.text1,
                pos_tags=pos1[:10],
                chunks=chunks1
            ),
            text2_processed=ProcessedResult(
                original=request.text2,
                pos_tags=pos2[:10],
                chunks=chunks2
            ),
            model_used=resolve_model_path(request.model),
            load_time=round(load_time, 2),
            inference_time=round(inference_time, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare/models", response_model=MultiModelResponse)
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
                "load_time": 0,
                "inference_time": 0,
                "success": False,
                "error": str(e)
            })
    
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return MultiModelResponse(
        results=results,
        text1=request.text1,
        text2=request.text2
    )

@app.post("/compare/matrix", response_model=MatrixResponse)
def compare_matrix(request: MatrixRequest):
    """여러 텍스트 유사도 매트릭스"""
    try:
        (tokenizer, model), _ = load_model(request.model)
        
        embeddings = [embed_text(t, tokenizer, model) for t in request.texts]
        
        n = len(request.texts)
        matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                matrix[i][j] = round(calculate_similarity(embeddings[i], embeddings[j]), 4)
        
        return MatrixResponse(
            similarity_matrix=matrix,
            texts=request.texts,
            model_used=resolve_model_path(request.model)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/models/cache")
def clear_model_cache():
    """모델 캐시 클리어"""
    global loaded_models, loaded_llms
    count = len(loaded_models) + len(loaded_llms)
    loaded_models = {}
    loaded_llms = {}
    torch.cuda.empty_cache()
    return {"message": f"Cleared {count} models from cache"}


# ═══════════════════════════════════════════════════════════════════════════
# RAG 엔드포인트
# ═══════════════════════════════════════════════════════════════════════════

class SearchRequest(BaseModel):
    query: str
    collection: str = "documents"
    n_results: int = 5
    model: str = "ko-sroberta"

class AskLLMRequest(BaseModel):
    query: str
    collection: str = "documents"
    n_results: int = 5
    embedding_model: str = "ko-sroberta"
    # 기본값을 가벼운 모델로 변경
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

class AskChunkRequest(BaseModel):
    """개별 청크에 대한 AI 답변 요청"""
    query: str
    chunk_text: str
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

class AddTextRequest(BaseModel):
    text: str
    doc_name: str = "manual_input"
    collection: str = "documents"
    model: str = "ko-sroberta"

class DeleteDocRequest(BaseModel):
    doc_name: str
    collection: str = "documents"


@app.post("/rag/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form(default="documents"),
    chunk_size: int = Form(default=500),
    chunk_method: str = Form(default="sentence"),
    overlap: int = Form(default=50),
    model: str = Form(default="ko-sroberta")
):
    """PDF/DOCX/TXT 파일 업로드 및 임베딩 저장"""
    try:
        content = await file.read()
        filename = file.filename
        model_path = resolve_model_path(model)
        
        text = load_document(filename, content)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="문서에서 텍스트를 추출할 수 없습니다.")
        
        chunks = create_chunks(text, chunk_size=chunk_size, overlap=overlap, method=chunk_method)
        chunk_texts = [c.text for c in chunks]
        
        # 메타데이터에 청킹 정보 포함
        metadata_list = []
        for i, c in enumerate(chunks):
            metadata_list.append({
                "doc_name": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_method": chunk_method,
                "chunk_size": chunk_size
            })
        
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/add-text")
def add_text_to_collection(request: AddTextRequest):
    """텍스트 직접 추가"""
    try:
        chunks = create_chunks(request.text, chunk_size=500, overlap=50)
        chunk_texts = [c.text for c in chunks]
        
        result = vector_store.add_documents(
            chunks=chunk_texts,
            doc_name=request.doc_name,
            collection_name=request.collection
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/search")
def search_documents(request: SearchRequest):
    """유사 문서 검색"""
    try:
        model_path = resolve_model_path(request.model)
        
        results = vector_store.search(
            query=request.query,
            collection_name=request.collection,
            n_results=request.n_results,
            model_name=model_path
        )
        
        return {
            "query": request.query,
            "results": results,
            "count": len(results),
            "model_used": model_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/ask")
def ask_question(request: SearchRequest):
    """질문에 대한 관련 문서 검색 + 컨텍스트 반환"""
    try:
        model_path = resolve_model_path(request.model)
        
        results, context = vector_store.search_with_context(
            query=request.query,
            collection_name=request.collection,
            n_results=request.n_results,
            model_name=model_path
        )
        
        return {
            "query": request.query,
            "context": context,
            "results": results,
            "count": len(results),
            "model_used": model_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/ask-llm")
def ask_with_llm(request: AskLLMRequest):
    """전체 컨텍스트 기반 LLM 답변"""
    try:
        embedding_model_path = resolve_model_path(request.embedding_model)

        results, context = vector_store.search_with_context(
            query=request.query,
            collection_name=request.collection,
            n_results=request.n_results,
            model_name=embedding_model_path
        )

        if not context.strip():
            return {
                "query": request.query,
                "answer": "관련 문서를 찾을 수 없습니다.",
                "sources": [],
            }

        tokenizer, model = load_llm_model(request.llm_model)

        # 컨텍스트를 토큰 기준으로 자르기
        MAX_CONTEXT_TOKENS = 700  # 프롬프트 + 질문 여유분 고려
        context_tokens = tokenizer.encode(context, add_special_tokens=False)
        
        if len(context_tokens) > MAX_CONTEXT_TOKENS:
            context_tokens = context_tokens[:MAX_CONTEXT_TOKENS]
            context = tokenizer.decode(context_tokens, skip_special_tokens=True)
            print(f"⚠️ 컨텍스트를 {MAX_CONTEXT_TOKENS} 토큰으로 잘랐습니다.")

        prompt = build_rag_prompt(
            query=request.query,
            context=context,
            language="ko"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024  # 최종 입력 1024 토큰 제한
        ).to(device)
        
        print(f"📝 입력 토큰 수: {inputs['input_ids'].shape[1]}")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                repetition_penalty=1.15,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        )

        answer = decoded.strip()

        return {
            "query": request.query,
            "answer": answer,
            "sources": results,
            "embedding_model": embedding_model_path,
            "llm_model": request.llm_model
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/ask-chunk")
def ask_with_single_chunk(request: AskChunkRequest):
    """개별 청크에 대한 LLM 답변"""
    try:
        prompt = build_chunk_prompt(
            query=request.query,
            chunk_text=request.chunk_text,
            language="ko"
        )

        tokenizer, model = load_llm_model(request.llm_model)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True,
                repetition_penalty=1.15,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        )

        answer = decoded.strip()

        return {
            "query": request.query,
            "answer": answer,
            "llm_model": request.llm_model
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/collections")
def list_collections():
    """컬렉션 목록"""
    try:
        collections = vector_store.list_collections()
        collection_info = []
        
        for name in collections:
            info = vector_store.get_collection_info(name)
            collection_info.append(info)
        
        return {"collections": collection_info}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/documents")
def list_documents(collection: str = "documents"):
    """저장된 문서 목록"""
    try:
        docs = vector_store.list_documents(collection)
        return {"documents": docs, "collection": collection}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/rag/document")
def delete_document(request: DeleteDocRequest):
    """문서 삭제"""
    try:
        result = vector_store.delete_by_doc_name(
            doc_name=request.doc_name,
            collection_name=request.collection
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/rag/collection/{collection_name}")
def delete_collection(collection_name: str):
    """컬렉션 전체 삭제"""
    try:
        result = vector_store.delete_all(collection_name)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/supported-formats")
def get_supported_formats():
    """지원하는 파일 형식"""
    return {"supported_extensions": get_supported_extensions()}


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
    
    print(f"\n🚀 Device: {device.upper()}")
    print("=" * 60)
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         텍스트 유사도 + RAG API 서버 v3.0                     ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  URL: http://localhost:8000                                   ║
    ║  Docs: http://localhost:8000/docs                             ║
    ║                                                               ║
    ║  📄 RAG 엔드포인트:                                           ║
    ║     POST /rag/upload      - 문서 업로드                       ║
    ║     POST /rag/search      - 검색                              ║
    ║     POST /rag/ask-llm     - 전체 AI 답변                      ║
    ║     POST /rag/ask-chunk   - 개별 청크 AI 답변                 ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000)