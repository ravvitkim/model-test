# 🔍 텍스트 유사도 비교 - 모델 벤치마크 v2

HuggingFace 임베딩 모델들을 비교할 수 있는 도구

```
[원문] → [파싱: 품사분석] → [청킹: 의미단위] → [임베딩: 벡터] → [코사인 유사도]
```

## ✨ 주요 기능

- **커스텀 모델 지원**: HuggingFace 모델 경로 직접 입력 가능
- **다중 모델 비교**: 여러 모델을 동시에 돌려서 유사도 비교
- **성능 측정**: 모델 로드 시간, 추론 시간 표시

---

## 📦 프리셋 모델

| 구분 | 키 | 모델 |
|------|-----|------|
| 🇰🇷 한국어 | `ko-sroberta` | jhgan/ko-sroberta-multitask |
| 🇰🇷 한국어 | `ko-sbert` | snunlp/KR-SBERT-V40K-klueNLI-augSTS |
| 🇰🇷 한국어 | `ko-simcse` | BM-K/KoSimCSE-roberta |
| 🌍 다국어 | `qwen3-0.6b` | Qwen/Qwen3-Embedding-0.6B |
| 🌍 다국어 | `qwen3-4b` | Qwen/Qwen3-Embedding-4B |
| 🌍 다국어 | `bge-m3` | BAAI/bge-m3 |
| 🌍 다국어 | `multilingual-e5` | intfloat/multilingual-e5-large |
| 🇺🇸 영어 | `mpnet` | sentence-transformers/all-mpnet-base-v2 |

**커스텀 모델 예시:**
- `intfloat/multilingual-e5-small`
- `BAAI/bge-base-en-v1.5`
- `Alibaba-NLP/gte-large-en-v1.5`

---

## 🚀 설치 순서

### 1. Backend (터미널 1)

```bash
# Conda 환경 생성
conda create -n similarity python=3.10 -y
conda activate similarity

# PyTorch 설치 (CUDA 12.6)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 나머지 패키지 설치
pip install -r requirements.txt

# 서버 실행
python main.py
```

### 2. Frontend (터미널 2)

```bash
cd frontend
npm install
npm run dev
```

---

## 🌐 접속

| 서비스 | URL |
|--------|-----|
| 프론트엔드 | http://localhost:3000 |
| API Docs | http://localhost:8000/docs |

---

## 📡 API 사용법

### 두 텍스트 비교 (프리셋 모델)
```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"text1": "인공지능은 미래다", "text2": "AI는 미래 기술이다", "model": "ko-sroberta"}'
```

### 두 텍스트 비교 (커스텀 모델)
```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"text1": "인공지능은 미래다", "text2": "AI는 미래 기술이다", "model": "Qwen/Qwen3-Embedding-0.6B"}'
```

### 여러 모델로 비교
```bash
curl -X POST http://localhost:8000/compare/models \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "인공지능은 미래다",
    "text2": "AI는 미래 기술이다",
    "models": ["ko-sroberta", "qwen3-0.6b", "Qwen/Qwen3-Embedding-0.6B"]
  }'
```

### 모델 캐시 클리어 (메모리 해제)
```bash
curl -X DELETE http://localhost:8000/models/cache
```

---

## 📁 프로젝트 구조

```
text-similarity-v2/
├── main.py              # FastAPI 백엔드
├── requirements.txt     # Python 패키지
├── README.md
└── frontend/
    ├── src/
    │   ├── App.tsx      # React (CSS 포함)
    │   └── main.tsx
    ├── package.json
    ├── index.html
    └── vite.config.ts
```

---

## ⚠️ CUDA 버전별 PyTorch

```bash
# CUDA 버전 확인
nvidia-smi

# CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 💡 팁

- 모델 첫 로드 시 다운로드가 필요해서 시간이 걸림
- 한번 로드된 모델은 캐싱되어 빠름
- 메모리 부족 시 `/models/cache` DELETE로 캐시 클리어