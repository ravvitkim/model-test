# RAG 시스템 리팩토링 v5.1

## 🔍 상위 5개 선정 기준 (원본 코드 분석)

### ChromaDB 검색 흐름
```
1. 쿼리 텍스트 → 임베딩 벡터 변환
2. ChromaDB collection.query() 호출
3. 코사인 거리(distance) 기준 정렬
4. 상위 N개 반환
5. similarity = 1 - distance 로 유사도 변환
```

**핵심 코드** (`vector_store.py` 490-510줄):
```python
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=min(n_results, collection.count()),  # 요청한 개수
    include=["documents", "metadatas", "distances"]
)

# distance → similarity 변환
similarity = 1 - distance
```

---

## ❓ 왜 5개 중 일부만 괜찮은 결과인가?

### 문제 1: 유사도 threshold 없음
- 0.2, 0.3 같은 낮은 유사도도 그냥 반환
- 실제로 "관련 없음" 수준의 결과가 포함됨

### 문제 2: 청킹이 제대로 작동 안 함 (핵심!)
```python
# main.py 원본 - 문제의 코드
@dataclass
class ParsedDocument:  # 중복 정의!
    text: str
    blocks: List[Dict] = field(default_factory=list)  # 항상 빈 리스트

def parse_document(...):
    return ParsedDocument(text=text, blocks=[], ...)  # blocks가 항상 []
```

결과적으로 `create_chunks_from_blocks()`가 빈 블록을 받아서 **청크가 전혀 생성되지 않는 경우가 있음**.

### 문제 3: 청크 크기 500자 (한국어에 비적합)
- 한국어 SOP 문서는 200-300자가 더 정확한 검색 결과

### 문제 4: 품질 지표 부재
- 어떤 결과가 신뢰할 만한지 알 수 없음

---

## ✅ v5.1 개선 사항

### 1. 검색 결과에 신뢰도(confidence) 추가

```python
# 새로운 유사도 기준
HIGH_CONFIDENCE_THRESHOLD = 0.65   # 🟢 high
DEFAULT_SIMILARITY_THRESHOLD = 0.35  # 🟡 medium
# 0.35 미만 = 🔴 low

# 검색 결과 예시
{
    "text": "제5조 ...",
    "similarity": 0.72,
    "confidence": "high",       # 새로 추가!
    "confidence_text": "🟢 높음 (신뢰할 수 있음)"
}
```

### 2. 유사도 threshold 필터링

```python
# API 요청 시 threshold 지정 가능
POST /rag/search
{
    "query": "품질 관리 절차",
    "similarity_threshold": 0.4  # 0.4 미만 결과 제외
}
```

### 3. 청킹 정상 작동

```python
# 수정된 코드 - document_loader.load_document() 직접 사용
parsed_doc = load_document(filename, content)  # blocks 정상 생성!

# 블록 기반 청킹
chunks = create_chunks_from_blocks(parsed_doc, ...)
```

### 4. 청크 크기 기본값 변경

```python
DEFAULT_CHUNK_SIZE = 300   # 기존 500 → 300 (한국어 최적화)
```

### 5. 품질 요약 제공

```python
# 검색 응답에 품질 요약 포함
{
    "results": [...],
    "quality_summary": {
        "avg_similarity": 0.58,
        "max_similarity": 0.72,
        "high_confidence_count": 2,  # 5개 중 2개만 신뢰도 높음
        "threshold_used": 0.35
    }
}
```

---

## 📁 수정된 파일

| 파일 | 주요 변경 |
|------|----------|
| `vector_store.py` | 유사도 threshold, confidence 분류, 품질 메트릭 |
| `main.py` | `load_document()` 직접 사용, 청크 크기 300, 품질 경고 |
| `document_loader.py` | `ContentBlock` import 추가, 조항 문서 자동 감지 |

---

## 🚀 사용 예시

### 고품질 결과만 보기
```bash
curl -X POST "http://localhost:8000/rag/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "품질 관리 절차",
    "similarity_threshold": 0.5,
    "n_results": 10
  }'
```

### 고급 검색 (상세 메트릭)
```bash
curl -X POST "http://localhost:8000/rag/search/advanced" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "품질 관리 절차",
    "n_results": 5
  }'
```

응답 예시:
```json
{
  "results": [
    {
      "text": "제5조 품질관리 절차...",
      "similarity": 0.72,
      "confidence": "high"
    },
    {
      "text": "품질 기준 정의...",
      "similarity": 0.58,
      "confidence": "medium"
    },
    {
      "text": "문서 관리...",
      "similarity": 0.31,
      "confidence": "low"  // ⚠️ 관련성 낮음
    }
  ],
  "quality_summary": {
    "high_count": 1,
    "medium_count": 1,
    "low_count": 1,
    "avg_similarity": 0.54
  }
}
```

---

## 💡 결론

**왜 5개 중 일부만 괜찮았나?**

1. **유사도 0.2~0.3 결과도 반환** → threshold 없었음
2. **청킹이 제대로 안 됨** → 블록이 빈 리스트로 생성
3. **품질 구분 없음** → 좋은 결과와 나쁜 결과 구분 불가

**해결책:**
- `confidence` 필드로 결과 품질 확인
- `similarity_threshold` 파라미터로 저품질 필터링
- 청킹 로직 수정으로 정확한 청크 생성
