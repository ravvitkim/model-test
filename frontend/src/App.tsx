import { useState, useRef } from 'react'
import './App.css'

// ═══════════════════════════════════════════════════════════════════════════
// 타입 정의
// ═══════════════════════════════════════════════════════════════════════════

interface ProcessedResult {
  original: string
  pos_tags: string[][]
  chunks: string[]
}

interface CompareResult {
  similarity: number
  interpretation: string
  text1_processed: ProcessedResult
  text2_processed: ProcessedResult
  model_used: string
  load_time: number
  inference_time: number
}

interface ModelResult {
  model_key: string
  model_path: string
  similarity: number
  interpretation: string
  load_time: number
  inference_time: number
  success: boolean
  error: string | null
}

interface MultiModelResult {
  results: ModelResult[]
  text1: string
  text2: string
}

interface MatrixResult {
  similarity_matrix: number[][]
  texts: string[]
  model_used: string
}

// RAG 타입
interface SearchResult {
  text: string
  similarity: number
  metadata: {
    doc_name: string
    chunk_index: number
    total_chunks?: number
    chunk_method?: string
    chunk_size?: number
  }
  aiAnswer?: string
  aiLoading?: boolean
}

interface RAGSearchResult {
  query: string
  results?: SearchResult[]
  sources?: SearchResult[]
  context?: string
  count?: number
  answer?: string
}

interface DocumentInfo {
  doc_name: string
  chunk_count: number
  chunk_method?: string
  chunk_size?: number
}

// ═══════════════════════════════════════════════════════════════════════════
// 프리셋 모델
// ═══════════════════════════════════════════════════════════════════════════

const PRESET_MODELS = [
  { key: 'ko-sroberta', name: 'Ko-SROBERTA', desc: '한국어 추천', category: 'korean' },
  { key: 'ko-sbert', name: 'Ko-SBERT', desc: '한국어', category: 'korean' },
  { key: 'ko-simcse', name: 'Ko-SimCSE', desc: '한국어', category: 'korean' },
  { key: 'qwen3-0.6b', name: 'Qwen3-0.6B', desc: '다국어 경량', category: 'multilingual' },
  { key: 'qwen3-4b', name: 'Qwen3-4B', desc: '다국어 고성능', category: 'multilingual' },
  { key: 'multilingual-minilm', name: 'MiniLM 다국어', desc: '경량', category: 'multilingual' },
  { key: 'multilingual-e5', name: 'E5 다국어', desc: '고성능', category: 'multilingual' },
  { key: 'bge-m3', name: 'BGE-M3', desc: '최신', category: 'multilingual' },
  { key: 'minilm', name: 'MiniLM', desc: '영어 경량', category: 'english' },
  { key: 'mpnet', name: 'MPNet', desc: '영어 고성능', category: 'english' },
]

const LLM_MODELS = [
  { key: 'Qwen/Qwen2.5-0.5B-Instruct', name: 'Qwen2.5-0.5B', desc: '초경량 (추천)' },
  { key: 'Qwen/Qwen2.5-1.5B-Instruct', name: 'Qwen2.5-1.5B', desc: '경량' },
  { key: 'Qwen/Qwen2.5-3B-Instruct', name: 'Qwen2.5-3B', desc: '고성능 (VRAM 6GB+)' },
  { key: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', name: 'TinyLlama', desc: '영어 특화' },
]

const API_URL = 'http://localhost:8000'

// ═══════════════════════════════════════════════════════════════════════════
// 유틸리티 함수
// ═══════════════════════════════════════════════════════════════════════════

const getSimilarityColor = (score: number) => {
  if (score >= 0.7) return '#22c55e'
  if (score >= 0.5) return '#eab308'
  if (score >= 0.3) return '#f97316'
  return '#ef4444'
}

const getSimilarityLabel = (score: number) => {
  if (score >= 0.8) return '매우 높음'
  if (score >= 0.6) return '높음'
  if (score >= 0.4) return '보통'
  if (score >= 0.2) return '낮음'
  return '매우 낮음'
}

// ═══════════════════════════════════════════════════════════════════════════
// 메인 컴포넌트
// ═══════════════════════════════════════════════════════════════════════════

function App() {
  const [text1, setText1] = useState('')
  const [text2, setText2] = useState('')
  const [selectedModel, setSelectedModel] = useState('ko-sroberta')
  const [result, setResult] = useState<CompareResult | null>(null)

  const [multiResult, setMultiResult] = useState<MultiModelResult | null>(null)
  const [selectedModels, setSelectedModels] = useState<string[]>(['ko-sroberta', 'qwen3-0.6b'])

  const [texts, setTexts] = useState<string[]>(['', '', ''])
  const [matrixResult, setMatrixResult] = useState<MatrixResult | null>(null)

  const [ragQuery, setRagQuery] = useState('')
  const [ragResult, setRagResult] = useState<RAGSearchResult | null>(null)
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [uploadStatus, setUploadStatus] = useState<string>('')
  const [ragModel, setRagModel] = useState('ko-sroberta')
  const [llmModel, setLlmModel] = useState('Qwen/Qwen2.5-3B-Instruct')
  const [chunkMethod, setChunkMethod] = useState<string>('sentence')
  const [chunkSize, setChunkSize] = useState<number>(300)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [globalAnswer, setGlobalAnswer] = useState<string>('')
  const [globalAnswerLoading, setGlobalAnswerLoading] = useState(false)

  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'single' | 'multi' | 'matrix' | 'rag'>('single')

  // 단일 모델 비교
  const handleCompare = async () => {
    if (!text1.trim() || !text2.trim()) {
      alert('두 텍스트를 모두 입력해주세요.')
      return
    }
    setLoading(true)
    setResult(null)
    try {
      const response = await fetch(`${API_URL}/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text1, text2, model: selectedModel })
      })
      if (!response.ok) throw new Error('API 요청 실패')
      setResult(await response.json())
    } catch (error) {
      alert('서버 연결 실패')
    } finally {
      setLoading(false)
    }
  }

  // 다중 모델 비교
  const handleMultiCompare = async () => {
    if (!text1.trim() || !text2.trim()) {
      alert('두 텍스트를 모두 입력해주세요.')
      return
    }
    if (selectedModels.length < 1) {
      alert('최소 1개 모델을 선택해주세요.')
      return
    }
    setLoading(true)
    setMultiResult(null)
    try {
      const response = await fetch(`${API_URL}/compare/models`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text1, text2, models: selectedModels })
      })
      if (!response.ok) throw new Error('API 요청 실패')
      setMultiResult(await response.json())
    } catch (error) {
      alert('서버 연결 실패')
    } finally {
      setLoading(false)
    }
  }

  // 매트릭스 비교
  const handleMatrixCompare = async () => {
    const validTexts = texts.filter(t => t.trim())
    if (validTexts.length < 2) {
      alert('최소 2개 텍스트를 입력해주세요.')
      return
    }
    setLoading(true)
    setMatrixResult(null)
    try {
      const response = await fetch(`${API_URL}/compare/matrix`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts: validTexts, model: selectedModel })
      })
      if (!response.ok) throw new Error('API 요청 실패')
      setMatrixResult(await response.json())
    } catch (error) {
      alert('서버 연결 실패')
    } finally {
      setLoading(false)
    }
  }

  // 파일 업로드
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setLoading(true)
    setUploadStatus('업로드 중...')
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('collection', 'documents')
      formData.append('chunk_size', chunkSize.toString())
      formData.append('chunk_method', chunkMethod)
      formData.append('model', ragModel)
      const response = await fetch(`${API_URL}/rag/upload`, {
        method: 'POST',
        body: formData
      })
      if (!response.ok) throw new Error('업로드 실패')
      const data = await response.json()
      setUploadStatus(`✅ ${data.filename} (${data.chunks_created}개 조각)`)
      fetchDocuments()
    } catch (error) {
      setUploadStatus(`❌ 업로드 실패`)
    } finally {
      setLoading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_URL}/rag/documents?collection=documents`)
      if (response.ok) {
        const data = await response.json()
        setDocuments(data.documents || [])
      }
    } catch (error) {
      console.error('문서 목록 로드 실패')
    }
  }

  // 검색만
  const handleRAGSearch = async () => {
    if (!ragQuery.trim()) {
      alert('질문을 입력해주세요.')
      return
    }
    setLoading(true)
    setRagResult(null)
    setGlobalAnswer('')
    try {
      const response = await fetch(`${API_URL}/rag/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: ragQuery,
          collection: 'documents',
          n_results: 5,
          model: ragModel
        })
      })
      if (!response.ok) throw new Error('검색 실패')
      setRagResult(await response.json())
    } catch (error) {
      alert('검색 실패')
    } finally {
      setLoading(false)
    }
  }

  // 전체 AI 답변
  const handleGlobalAIAnswer = async () => {
    if (!ragQuery.trim()) return
    setGlobalAnswerLoading(true)
    setGlobalAnswer('')
    try {
      const response = await fetch(`${API_URL}/rag/ask-llm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: ragQuery,
          collection: 'documents',
          n_results: 5,
          embedding_model: ragModel,
          llm_model: llmModel
        })
      })
      if (!response.ok) throw new Error('LLM 실패')
      const data = await response.json()
      setGlobalAnswer(data.answer || '답변 생성 실패')
      if (data.sources) {
        setRagResult(prev => prev ? { ...prev, results: data.sources } : { query: ragQuery, results: data.sources })
      }
    } catch (error) {
      setGlobalAnswer('오류 발생')
    } finally {
      setGlobalAnswerLoading(false)
    }
  }

  // 개별 청크 AI 답변
  const handleChunkAIAnswer = async (index: number, chunkText: string) => {
    if (!ragResult?.results) return
    const updatedResults = [...ragResult.results]
    updatedResults[index] = { ...updatedResults[index], aiLoading: true }
    setRagResult({ ...ragResult, results: updatedResults })

    try {
      const response = await fetch(`${API_URL}/rag/ask-chunk`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: ragQuery,
          chunk_text: chunkText,
          llm_model: llmModel
        })
      })
      if (!response.ok) throw new Error('실패')
      const data = await response.json()
      const newResults = [...(ragResult.results || [])]
      newResults[index] = { ...newResults[index], aiAnswer: data.answer || '답변 실패', aiLoading: false }
      setRagResult({ ...ragResult, results: newResults })
    } catch {
      const newResults = [...(ragResult.results || [])]
      newResults[index] = { ...newResults[index], aiAnswer: '오류 발생', aiLoading: false }
      setRagResult({ ...ragResult, results: newResults })
    }
  }

  const handleDeleteDocument = async (docName: string) => {
    if (!confirm(`"${docName}" 삭제?`)) return
    try {
      const response = await fetch(`${API_URL}/rag/document`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_name: docName, collection: 'documents' })
      })
      if (response.ok) {
        fetchDocuments()
        setUploadStatus(`🗑️ ${docName} 삭제됨`)
      }
    } catch {
      alert('삭제 실패')
    }
  }

  const toggleModelSelection = (modelKey: string) => {
    setSelectedModels(prev =>
      prev.includes(modelKey) ? prev.filter(m => m !== modelKey) : [...prev, modelKey]
    )
  }

  const updateText = (index: number, value: string) => {
    const newTexts = [...texts]
    newTexts[index] = value
    setTexts(newTexts)
  }

  const handleTabChange = (tab: 'single' | 'multi' | 'matrix' | 'rag') => {
    setActiveTab(tab)
    if (tab === 'rag') fetchDocuments()
  }

  return (
    <div className="app">
      <header className="header">
        <h1 className="title">🔍 텍스트 유사도 + RAG</h1>
        <p className="subtitle">문서 업로드 → 검색 → AI 답변</p>
      </header>

      <div className="tabs">
        {['single', 'multi', 'matrix', 'rag'].map(tab => (
          <button
            key={tab}
            className={`tab ${activeTab === tab ? 'active' : ''}`}
            onClick={() => handleTabChange(tab as any)}
          >
            {tab === 'single' && '단일 비교'}
            {tab === 'multi' && '모델 비교'}
            {tab === 'matrix' && '매트릭스'}
            {tab === 'rag' && '📄 RAG'}
          </button>
        ))}
      </div>

      <main className="main">
        {/* 단일 비교 */}
        {activeTab === 'single' && (
          <>
            <div className="input-section">
              <div className="text-input">
                <label>텍스트 1</label>
                <textarea value={text1} onChange={(e) => setText1(e.target.value)} placeholder="첫 번째 텍스트..." rows={5} />
              </div>
              <div className="text-input">
                <label>텍스트 2</label>
                <textarea value={text2} onChange={(e) => setText2(e.target.value)} placeholder="두 번째 텍스트..." rows={5} />
              </div>
            </div>
            <div className="model-select">
              <label>모델</label>
              <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                {PRESET_MODELS.map(m => <option key={m.key} value={m.key}>{m.name}</option>)}
              </select>
            </div>
            <button className="primary-btn" onClick={handleCompare} disabled={loading}>
              {loading ? '분석 중...' : '비교하기'}
            </button>
            {result && (
              <div className="result-box">
                <div className="score-big" style={{ color: getSimilarityColor(result.similarity) }}>
                  {(result.similarity * 100).toFixed(1)}%
                </div>
                <div className="score-label">{result.interpretation}</div>
                <div className="score-bar">
                  <div className="score-fill" style={{ width: `${result.similarity * 100}%`, backgroundColor: getSimilarityColor(result.similarity) }} />
                </div>
              </div>
            )}
          </>
        )}

        {/* 모델 비교 */}
        {activeTab === 'multi' && (
          <>
            <div className="input-section">
              <div className="text-input">
                <label>텍스트 1</label>
                <textarea value={text1} onChange={(e) => setText1(e.target.value)} placeholder="첫 번째 텍스트..." rows={4} />
              </div>
              <div className="text-input">
                <label>텍스트 2</label>
                <textarea value={text2} onChange={(e) => setText2(e.target.value)} placeholder="두 번째 텍스트..." rows={4} />
              </div>
            </div>
            <div className="model-grid">
              {PRESET_MODELS.map(m => (
                <label key={m.key} className={`model-chip ${selectedModels.includes(m.key) ? 'selected' : ''}`}>
                  <input type="checkbox" checked={selectedModels.includes(m.key)} onChange={() => toggleModelSelection(m.key)} />
                  {m.name}
                </label>
              ))}
            </div>
            <button className="primary-btn" onClick={handleMultiCompare} disabled={loading}>
              {loading ? '비교 중...' : `${selectedModels.length}개 모델로 비교`}
            </button>
            {multiResult && (
              <div className="results-list">
                {multiResult.results.map((r, i) => (
                  <div key={i} className="result-row">
                    <span className="result-name">{r.model_key}</span>
                    <span className="result-score" style={{ color: getSimilarityColor(r.similarity) }}>
                      {(r.similarity * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {/* 매트릭스 */}
        {activeTab === 'matrix' && (
          <>
            <div className="matrix-inputs">
              {texts.map((text, i) => (
                <div key={i} className="matrix-row">
                  <span className="row-num">{i + 1}</span>
                  <textarea value={text} onChange={(e) => updateText(i, e.target.value)} placeholder={`텍스트 ${i + 1}`} rows={2} />
                  {texts.length > 2 && (
                    <button className="remove-btn" onClick={() => setTexts(texts.filter((_, j) => j !== i))}>×</button>
                  )}
                </div>
              ))}
              {texts.length < 10 && (
                <button className="add-btn" onClick={() => setTexts([...texts, ''])}>+ 추가</button>
              )}
            </div>
            <button className="primary-btn" onClick={handleMatrixCompare} disabled={loading}>
              {loading ? '계산 중...' : '매트릭스 생성'}
            </button>
            {matrixResult && (
              <div className="matrix-table-wrap">
                <table className="matrix-table">
                  <thead>
                    <tr>
                      <th></th>
                      {matrixResult.texts.map((_, i) => <th key={i}>{i + 1}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {matrixResult.similarity_matrix.map((row, i) => (
                      <tr key={i}>
                        <td className="row-head">{i + 1}</td>
                        {row.map((score, j) => (
                          <td key={j} style={{ backgroundColor: i === j ? '#333' : `${getSimilarityColor(score)}33`, color: getSimilarityColor(score) }}>
                            {(score * 100).toFixed(0)}%
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}

        {/* RAG */}
        {activeTab === 'rag' && (
          <>
            {/* 설정 */}
            <div className="settings-row">
              <div className="setting">
                <label>🔍 검색 모델</label>
                <select value={ragModel} onChange={(e) => setRagModel(e.target.value)}>
                  {PRESET_MODELS.map(m => <option key={m.key} value={m.key}>{m.name}</option>)}
                </select>
              </div>
              <div className="setting">
                <label>🤖 답변 모델</label>
                <select value={llmModel} onChange={(e) => setLlmModel(e.target.value)}>
                  {LLM_MODELS.map(m => <option key={m.key} value={m.key}>{m.name}</option>)}
                </select>
              </div>
            </div>

            {/* 청킹 설정 */}
            <div className="chunk-settings">
              <div className="chunk-method">
                <button className={chunkMethod === 'sentence' ? 'active' : ''} onClick={() => setChunkMethod('sentence')}>
                  📝 문장 단위
                </button>
                <button className={chunkMethod === 'paragraph' ? 'active' : ''} onClick={() => setChunkMethod('paragraph')}>
                  📄 문단 단위
                </button>
              </div>
              <div className="chunk-size">
                <span>조각 크기: {chunkSize}자</span>
                <input type="range" min="200" max="2000" step="100" value={chunkSize} onChange={(e) => setChunkSize(Number(e.target.value))} />
              </div>
            </div>

            {/* 업로드 */}
            <div className="upload-section">
              <label>📁 문서 업로드 (PDF, DOCX, TXT)</label>
              <input ref={fileInputRef} type="file" accept=".pdf,.docx,.doc,.txt" onChange={handleFileUpload} disabled={loading} />
              {uploadStatus && <p className="status">{uploadStatus}</p>}
            </div>

            {/* 문서 목록 */}
            {documents.length > 0 && (
              <div className="doc-list">
                <label>📚 업로드된 문서</label>
                {documents.map((doc, i) => (
                  <div key={i} className="doc-item">
                    <div>
                      <strong>{doc.doc_name}</strong>
                      <span className="doc-meta">{doc.chunk_count}개 조각</span>
                    </div>
                    <button onClick={() => handleDeleteDocument(doc.doc_name)}>🗑️</button>
                  </div>
                ))}
              </div>
            )}

            {/* 질문 */}
            <div className="query-section">
              <label>💬 질문</label>
              <textarea value={ragQuery} onChange={(e) => setRagQuery(e.target.value)} placeholder="문서에 대해 질문하세요..." rows={3} />
              <div className="query-btns">
                <button className="search-btn" onClick={handleRAGSearch} disabled={loading || documents.length === 0}>
                  🔍 검색만
                </button>
                <button
                  className="ai-btn"
                  onClick={async () => { await handleRAGSearch(); await handleGlobalAIAnswer(); }}
                  disabled={loading || globalAnswerLoading || documents.length === 0}
                >
                  ✨ 검색 + AI 답변
                </button>
              </div>
            </div>

            {/* 전체 AI 답변 */}
            {(globalAnswerLoading || globalAnswer) && (
              <div className="global-answer">
                <h3>🤖 AI 종합 답변</h3>
                {globalAnswerLoading ? (
                  <div className="loading-answer">답변 생성 중...</div>
                ) : (
                  <div className="answer-text">{globalAnswer}</div>
                )}
              </div>
            )}

            {/* 검색 결과 */}
            {ragResult?.results && ragResult.results.length > 0 && (
              <div className="search-results">
                <h3>📄 관련 문서 조각 ({ragResult.results.length}개)</h3>
                
                {ragResult.results.map((r, idx) => (
                  <div key={idx} className="result-card">
                    {/* 상단: 출처 + 연관도 */}
                    <div className="card-header">
                      <span className="source-file">📄 {r.metadata?.doc_name}</span>
                      <div className="relevance" style={{ color: getSimilarityColor(r.similarity) }}>
                        <span className="relevance-value">{getSimilarityLabel(r.similarity)}</span>
                        <span className="relevance-percent">{(r.similarity * 100).toFixed(0)}%</span>
                      </div>
                    </div>

                    {/* 본문 */}
                    <div className="card-content">{r.text}</div>

                    {/* 개별 AI 버튼 */}
                    <button
                      className="chunk-ai-btn"
                      onClick={() => handleChunkAIAnswer(idx, r.text)}
                      disabled={r.aiLoading}
                    >
                      {r.aiLoading ? '생성 중...' : '🤖 이 내용으로 답변 생성'}
                    </button>

                    {/* 개별 AI 답변 */}
                    {r.aiAnswer && (
                      <div className="chunk-answer">
                        <div className="chunk-answer-title">💡 AI 답변</div>
                        <div className="chunk-answer-text">{r.aiAnswer}</div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {ragResult && (!ragResult.results || ragResult.results.length === 0) && !loading && (
              <div className="no-results">관련 문서를 찾을 수 없습니다.</div>
            )}
          </>
        )}
      </main>

      <footer className="footer">
        HuggingFace 임베딩 + LLM 기반 RAG 시스템
      </footer>
    </div>
  )
}

export default App