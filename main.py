"""
텍스트 유사도 비교 API - 커스텀 모델 지원
[원문] → [파싱: 품사 분석] → [청킹: 의미 단위] → [임베딩: 벡터 변환] → [코사인 유사도]
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re
import time

app = FastAPI(title="Text Similarity API", version="2.0.0")

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
device = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════════════════
# Pydantic 모델
# ═══════════════════════════════════════════════════════════════════════════

class CompareRequest(BaseModel):
    text1: str
    text2: str
    model: str = "ko-sroberta"  # 프리셋 키 또는 HuggingFace 모델 경로

class MultiModelCompareRequest(BaseModel):
    text1: str
    text2: str
    models: List[str]  # 프리셋 키 또는 HuggingFace 모델 경로 리스트

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
    """모델 로드 (캐싱) - 프리셋 키 또는 직접 HuggingFace 경로"""
    model_path = resolve_model_path(model_key)
    
    # 이미 로드됨
    if model_path in loaded_models:
        return loaded_models[model_path], 0.0
    
    print(f"📦 Loading model: {model_path}...")
    start_time = time.time()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        model.eval()
        
        load_time = time.time() - start_time
        loaded_models[model_path] = (tokenizer, model)
        print(f"✅ Model loaded: {model_path} ({load_time:.2f}s)")
        
        return (tokenizer, model), load_time
    except Exception as e:
        raise ValueError(f"모델 로드 실패: {model_path} - {str(e)}")

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
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean Pooling
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
        "message": "Text Similarity API v2.0",
        "endpoints": ["/compare", "/compare/models", "/compare/matrix", "/models"],
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

@app.get("/models")
def get_models():
    """사용 가능한 프리셋 모델 + 로드된 모델"""
    return {
        "preset_models": PRESET_MODELS,
        "loaded_models": list(loaded_models.keys()),
        "device": device,
        "tip": "프리셋 키 또는 HuggingFace 모델 경로 직접 입력 가능 (예: Qwen/Qwen3-Embedding-0.6B)"
    }

@app.post("/models/add")
def add_preset_model(request: AddModelRequest):
    """프리셋에 새 모델 추가"""
    PRESET_MODELS[request.key] = request.model_path
    return {"message": f"Added {request.key}: {request.model_path}", "presets": PRESET_MODELS}

@app.post("/compare", response_model=CompareResponse)
def compare_texts(request: CompareRequest):
    """두 텍스트 비교 - 프리셋 키 또는 HuggingFace 경로 사용 가능"""
    try:
        (tokenizer, model), load_time = load_model(request.model)
        
        start_time = time.time()
        
        # Stage 1: 품사 분석
        pos1 = parse_pos(request.text1, tokenizer)
        pos2 = parse_pos(request.text2, tokenizer)
        
        # Stage 2: 청킹
        chunks1 = chunk_text(request.text1)
        chunks2 = chunk_text(request.text2)
        
        # Stage 3: 임베딩
        emb1 = embed_text(request.text1, tokenizer, model)
        emb2 = embed_text(request.text2, tokenizer, model)
        
        # Stage 4: 코사인 유사도
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
    """여러 모델로 동시 비교 - 커스텀 모델 경로 지원"""
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
    
    # 유사도 높은 순 정렬
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
    """모델 캐시 클리어 (메모리 해제)"""
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
    
    # GPU/CPU 상태 체크
    print("\n" + "=" * 60)
    print("🖥️  시스템 정보")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA 사용 가능")
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        print(f"   - CUDA 버전: {torch.version.cuda}")
        print(f"   - PyTorch 버전: {torch.__version__}")
        print(f"   - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ CUDA 사용 불가 - CPU 모드로 실행")
        print(f"   - PyTorch 버전: {torch.__version__}")
    
    print(f"\n🚀 Device: {device.upper()}")
    print("=" * 60)
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║         텍스트 유사도 API 서버 v2.0                               ║
    ║  [원문] → [파싱] → [청킹] → [임베딩] → [코사인 유사도]            ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  URL: http://localhost:8000                                       ║
    ║  Docs: http://localhost:8000/docs                                 ║
    ║                                                                   ║
    ║  💡 커스텀 모델 사용법:                                           ║
    ║     model: "Qwen/Qwen3-Embedding-0.6B"                           ║
    ║     model: "intfloat/multilingual-e5-small"                      ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000)