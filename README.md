# RAG Chatbot v6.2 - SOP 문서 챗봇

## 🎯 주요 기능

### 1. **section_path 계층 추적** (v6.2 신규)
각 청크에 조항의 계층 경로가 자동 생성됩니다:

```json
{
  "text": "품질경영매뉴얼은 회사 전반에 적용되는 최상위 문서이다...",
  "metadata": {
    "doc_title": "GMP 문서 체계",
    "sop_id": "EQ-SOP-00001",
    "section": "5.1.1",
    "section_path": "5 > 5.1 > 5.1.1",
    "section_path_readable": "5 절차 > 5.1 문서체계 > 5.1.1 Level 1 (품질매뉴얼)",
    "title": "Level 1 (품질매뉴얼)"
  }
}
```

### 2. **챗봇 인터페이스**
- 대화 히스토리 유지
- 세션 관리
- 친근한 응답 스타일

### 3. **RAG 기반 답변**
- 문서 검색 + LLM 답변 생성
- 출처 표시 (section_path 포함)
- 유사도 점수 표시

## 📁 프로젝트 구조

```
rag_chatbot/
├── main.py                 # FastAPI 서버
├── requirements.txt        # Python 의존성
├── rag/                    # RAG 모듈
│   ├── __init__.py
│   ├── document_loader.py  # 🔥 section_path 추가됨
│   ├── chunker.py          # 🔥 section_path 전달
│   ├── vector_store.py     # ChromaDB
│   ├── llm.py              # Ollama/HuggingFace
│   └── prompt.py           # 프롬프트 템플릿
└── frontend/               # React 프론트엔드
    ├── package.json
    ├── vite.config.ts
    └── src/
        ├── App.tsx         # 챗봇 UI
        └── App.css         # 스타일
```

## 🚀 실행 방법

### 1. 백엔드 (FastAPI)

```bash
cd rag_chatbot

# 의존성 설치
pip install -r requirements.txt --break-system-packages

# 서버 실행
python main.py
```

서버: http://localhost:8000
API 문서: http://localhost:8000/docs

### 2. 프론트엔드 (React)

```bash
cd frontend

# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

프론트엔드: http://localhost:5173

### 3. Ollama (선택사항)

```bash
# Ollama 설치 후
ollama serve

# 모델 다운로드
ollama pull qwen2.5:3b
```

## 📝 API 엔드포인트

### 챗봇

```bash
# 채팅
POST /chat
{
  "message": "품질매뉴얼이 뭐야?",
  "session_id": "optional-session-id",
  "embedding_model": "multilingual-e5-small",
  "llm_model": "qwen2.5:3b"
}

# 대화 히스토리 조회
GET /chat/history/{session_id}

# 대화 초기화
DELETE /chat/history/{session_id}
```

### 문서 관리

```bash
# 문서 업로드
POST /rag/upload
# FormData: file, collection, chunk_method, model

# 문서 목록
GET /rag/documents

# 문서 삭제
DELETE /rag/document
```

### 검색

```bash
# 검색
POST /rag/search
{
  "query": "품질매뉴얼",
  "n_results": 5
}

# RAG 답변
POST /rag/ask
{
  "query": "품질매뉴얼이란?",
  "embedding_model": "multilingual-e5-small",
  "llm_model": "qwen2.5:3b"
}
```

## 🔧 section_path 동작 원리

### 1. 조항 패턴 인식

```python
ARTICLE_PATTERNS = [
    (r'^(\d+)\.\s+([가-힣A-Za-z].+)', 'section'),           # "5. 절차"
    (r'^(\d+\.\d+)\s+([가-힣A-Za-z].+)', 'subsection'),     # "5.1 문서체계"
    (r'^(\d+\.\d+\.\d+)\s+([가-힣A-Za-z].+)', 'subsubsection'), # "5.1.1 Level 1"
]
```

### 2. 스택 기반 추적

```python
section_stack = {
    "section": {"num": "5", "title": "절차"},
    "subsection": {"num": "5.1", "title": "문서체계"},
    "subsubsection": {"num": "5.1.1", "title": "Level 1"}
}
```

### 3. 경로 생성

```python
section_path = "5 > 5.1 > 5.1.1"
section_path_readable = "5 절차 > 5.1 문서체계 > 5.1.1 Level 1"
```

## ⚙️ 설정 옵션

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `embedding_model` | `multilingual-e5-small` | 임베딩 모델 |
| `llm_model` | `qwen2.5:3b` | LLM 모델 |
| `chunk_method` | `article` | 청킹 방식 |
| `chunk_size` | `500` | 청크 크기 |
| `similarity_threshold` | `0.35` | 유사도 임계값 |

## 📌 v6.2 변경사항

1. **document_loader.py**
   - `_extract_article_blocks()` 함수에 section_stack 추가
   - `build_section_path()` 함수로 계층 경로 생성
   - `section_path`, `section_path_readable` 메타데이터 추가

2. **chunker.py**
   - `create_chunks_from_blocks()`에서 section_path 전달

3. **main.py**
   - `/chat` 엔드포인트 추가 (대화 히스토리 지원)
   - `format_metadata_display()`에서 section_path 표시

4. **App.tsx**
   - 챗봇 UI로 변경
   - section_path 시각적 표시
   - 대화 히스토리 관리