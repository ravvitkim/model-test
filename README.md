# 🔍 텍스트 유사도 + RAG 시스템 v4.0

조항 단위 청킹 + 에이전트 패턴(되묻기) + Ollama 지원

## ✨ 주요 변경사항 (v4.0)

### 1. 조항 단위 청킹 (SOP/법률 문서용)
- `제1조`, `제2조`, `1.`, `가.`, `①` 등 조항 패턴 자동 인식
- 긴 조항은 설정된 크기로 분할
- 메타데이터: 문서명, 제목, 섹션, 조항 번호

### 2. 에이전트 패턴 (되묻기)
- 여러 문서에서 유사한 점수로 결과가 나오면 사용자에게 되묻기
- 예: "손 씻는 방법" → "어떤 SOP의 손 씻는 방법을 원하시나요?"

### 3. Ollama 지원 (로컬 LLM)
- Quantized 모델로 적은 VRAM으로 큰 모델 사용 가능
- 권장: `qwen2.5:3b` (3GB VRAM)

## 🚀 설치

### 1. Python 패키지
```bash
conda create -n rag python=3.10 -y
conda activate rag

# PyTorch (CUDA 버전에 맞게)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 나머지 패키지
pip install -r requirements.txt
```

### 2. Ollama 설치 (권장)
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# 모델 다운로드
ollama pull qwen2.5:3b   # 3GB, 추천
ollama pull qwen2.5:7b   # 5GB, 고성능

# 서버 시작
ollama serve
```

### 3. 서버 실행
```bash
python main.py
```

## 📡 API 엔드포인트

### RAG
| 엔드포인트 | 설명 |
|------------|------|
| `POST /rag/upload` | 문서 업로드 |
| `GET /rag/documents` | 문서 목록 |
| `POST /rag/search` | 벡터 검색 |
| `POST /rag/ask` | **RAG 질의응답 (되묻기 포함)** |
| `POST /rag/ask-llm` | RAG 질의응답 (되묻기 없이) |
| `POST /rag/ask-chunk` | 단일 청크 답변 |

### 시스템
| 엔드포인트 | 설명 |
|------------|------|
| `GET /models/llm` | LLM 모델 목록 (Ollama 상태) |
| `DELETE /models/cache` | 모델 캐시 클리어 |

## 🔧 청킹 방법

| 방법 | 설명 | 용도 |
|------|------|------|
| `article` | 조항 단위 (제1조, 1. 등) | **SOP, 법률 문서 (기본값)** |
| `sentence` | 문장 단위 | 일반 문서 |
| `paragraph` | 문단 단위 | 긴 문서 |

## 🤖 LLM 모델

### Ollama (로컬 추천)
| 모델 | VRAM | 설명 |
|------|------|------|
| `qwen2.5:0.5b` | 1GB | 초경량 |
| `qwen2.5:3b` | 3GB | **추천** |
| `qwen2.5:7b` | 5GB | 고성능 |

### HuggingFace
| 모델 | 설명 |
|------|------|
| `Qwen/Qwen2.5-0.5B-Instruct` | 초경량 |
| `Qwen/Qwen2.5-3B-Instruct` | VRAM 6GB+ |

## 💡 에이전트 패턴 (되묻기)

```python
# 여러 문서에서 비슷한 점수로 결과가 나오면 되묻기
POST /rag/ask
{
    "query": "손 씻는 방법을 알려주세요",
    "llm_model": "qwen2.5:3b",
    "llm_backend": "ollama",
    "check_clarification": true
}

# 응답 (되묻기 필요 시)
{
    "answer": "손 씻는 방법에 대해 여러 SOP에서...",
    "needs_clarification": true,
    "clarification_options": ["SOP-001.pdf", "SOP-002.pdf"]
}

# 특정 문서 선택 후 재요청
POST /rag/ask
{
    "query": "손 씻는 방법을 알려주세요",
    "filter_doc": "SOP-001.pdf",
    "check_clarification": false
}
```

## 📁 프로젝트 구조

```
rag/
├── main.py              # FastAPI 서버
├── chunker.py           # 청킹 (조항/문장/문단)
├── document_loader.py   # PDF/DOCX/TXT 로더
├── vector_store.py      # ChromaDB 벡터 스토어
├── llm.py               # LLM (Ollama + HuggingFace)
├── prompt.py            # 프롬프트 템플릿
└── requirements.txt
```