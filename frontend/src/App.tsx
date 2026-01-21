import { useState, useRef, useEffect } from 'react'
import './App.css'

// ═══════════════════════════════════════════════════════════════════════════
// 타입 정의
// ═══════════════════════════════════════════════════════════════════════════

interface MetadataDisplay {
  doc_name?: string
  doc_title?: string
  sop_id?: string
  version?: string
  section?: string  // 제N조 형식
  title?: string
  page?: string
}

interface SearchResult {
  text: string
  similarity: number
  interpretation?: string
  confidence?: string
  confidence_text?: string
  metadata: Record<string, any>
  metadata_display?: MetadataDisplay  // 가독성 개선된 메타데이터
  aiAnswer?: string
  aiLoading?: boolean
}

interface ClarificationOption {
  doc_name: string
  doc_title?: string
  display_text: string
  score: number
  sections?: string[]
}

interface RAGResponse {
  query: string
  answer?: string
  results?: SearchResult[]
  sources?: SearchResult[]
  needs_clarification?: boolean
  clarification_options?: ClarificationOption[]
  action_taken?: string
  agent_type?: string
}

interface DocumentInfo {
  doc_name: string
  doc_title?: string
  chunk_count: number
  chunk_method?: string
}

interface EmbeddingModelSpec {
  path: string
  name: string
  dim: number
  memory_mb: number
  lang: string
  compatible: boolean
}

interface EmbeddingModelsResponse {
  all: EmbeddingModelSpec[]
  compatible: EmbeddingModelSpec[]
  incompatible: EmbeddingModelSpec[]
  filter_criteria: { max_dim: number; max_memory_mb: number }
}

// ═══════════════════════════════════════════════════════════════════════════
// 프리셋
// ═══════════════════════════════════════════════════════════════════════════

const PRESET_MODELS = [
  { key: 'multilingual-e5-small', name: 'E5-Small 다국어', desc: '경량 추천 ⭐', dim: 384, mem: 120, compatible: true },
  { key: 'ko-sroberta', name: 'Ko-SROBERTA', desc: '한국어', dim: 768, mem: 440, compatible: true },
  { key: 'ko-sbert', name: 'Ko-SBERT', desc: '한국어', dim: 768, mem: 440, compatible: true },
  { key: 'ko-simcse', name: 'Ko-SimCSE', desc: '한국어', dim: 768, mem: 440, compatible: true },
  { key: 'multilingual-minilm', name: 'MiniLM 다국어', desc: '경량', dim: 384, mem: 470, compatible: true },
  { key: 'multilingual-e5-large', name: 'E5-Large 다국어', desc: '고성능', dim: 1024, mem: 1200, compatible: true },
  { key: 'bge-m3', name: 'BGE-M3', desc: '고성능', dim: 1024, mem: 1300, compatible: true },
  { key: 'minilm', name: 'MiniLM', desc: '영어 경량', dim: 384, mem: 90, compatible: true },
  { key: 'mpnet', name: 'MPNet', desc: '영어', dim: 768, mem: 420, compatible: true },
  { key: 'qwen3-0.6b', name: 'Qwen3-0.6B', desc: '다국어', dim: 1024, mem: 600, compatible: true },
]

const OLLAMA_MODELS = [
  { key: 'qwen2.5:0.5b', name: 'Qwen2.5-0.5B', desc: '초경량 (1GB)' },
  { key: 'qwen2.5:1.5b', name: 'Qwen2.5-1.5B', desc: '경량 (2GB)' },
  { key: 'qwen2.5:3b', name: 'Qwen2.5-3B', desc: '추천 (3GB)' },
  { key: 'qwen3:4b', name: 'Qwen3-4B', desc: '최신 (4GB)' },
  { key: 'llama3.2:3b', name: 'Llama3.2-3B', desc: '경량 (3GB)' },
]

const HF_MODELS = [
  { key: 'Qwen/Qwen2.5-0.5B-Instruct', name: 'Qwen2.5-0.5B', desc: '초경량' },
  { key: 'Qwen/Qwen2.5-1.5B-Instruct', name: 'Qwen2.5-1.5B', desc: '경량' },
]

const CHUNK_METHODS = [
  { key: 'article', name: '📜 조항 단위', desc: 'SOP/법률 권장' },
  { key: 'recursive', name: '🔄 Recursive', desc: '랭체인 스타일' },
  { key: 'semantic', name: '🧠 Semantic', desc: '의미 기반' },
  { key: 'sentence', name: '📝 문장 단위', desc: '빠름' },
  { key: 'paragraph', name: '📄 문단 단위', desc: '중간' },
  { key: 'llm', name: '🤖 LLM 파싱', desc: '가장 정교' },
]

const AGENT_TYPES = [
  { key: 'basic', name: '기본', desc: '단순 검색+답변' },
  { key: 'react', name: 'ReAct', desc: 'Reasoning+Acting' },
  { key: 'plan_execute', name: 'Plan & Execute', desc: '계획 후 실행' },
]

const API_URL = 'http://localhost:8000'

// ═══════════════════════════════════════════════════════════════════════════
// 유틸리티
// ═══════════════════════════════════════════════════════════════════════════

const getSimilarityColor = (score: number) => {
  if (score >= 0.7) return '#22c55e'
  if (score >= 0.5) return '#eab308'
  if (score >= 0.3) return '#f97316'
  return '#ef4444'
}

const getConfidenceColor = (confidence?: string) => {
  if (confidence === 'high') return '#22c55e'
  if (confidence === 'medium') return '#eab308'
  return '#ef4444'
}

// ═══════════════════════════════════════════════════════════════════════════
// 메인 컴포넌트
// ═══════════════════════════════════════════════════════════════════════════

function App() {
  // 상태
  const [ragQuery, setRagQuery] = useState('')
  const [ragResult, setRagResult] = useState<RAGResponse | null>(null)
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [uploadStatus, setUploadStatus] = useState('')
  const [loading, setLoading] = useState(false)

  // 모델 설정
  const [ragModel, setRagModel] = useState('multilingual-e5-small')
  const [llmBackend, setLlmBackend] = useState<'ollama' | 'huggingface'>('ollama')
  const [llmModel, setLlmModel] = useState('qwen2.5:3b')
  const [ollamaStatus, setOllamaStatus] = useState<{ running: boolean; models: string[] }>({ running: false, models: [] })

  // 청킹 설정
  const [chunkMethod, setChunkMethod] = useState('article')
  const [chunkSize, setChunkSize] = useState(500)
  const [semanticThreshold, setSemanticThreshold] = useState(0.5)

  // 에이전트 설정
  const [useAgent, setUseAgent] = useState(false)
  const [agentType, setAgentType] = useState('basic')
  const [enableClarification, setEnableClarification] = useState(true)

  // UI 상태
  const [clarificationMessage, setClarificationMessage] = useState<string | null>(null)
  const [clarificationOptions, setClarificationOptions] = useState<ClarificationOption[]>([])
  const [globalAnswer, setGlobalAnswer] = useState('')
  const [globalAnswerLoading, setGlobalAnswerLoading] = useState(false)
  const [expandedMeta, setExpandedMeta] = useState<Set<number>>(new Set())
  const [embeddingModels, setEmbeddingModels] = useState<EmbeddingModelsResponse | null>(null)
  const [showModelInfo, setShowModelInfo] = useState(false)

  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    checkOllamaStatus()
    fetchDocuments()
    fetchEmbeddingModels()
  }, [])

  // ─────────────────────────────────────────────────────────────
  // API 호출
  // ─────────────────────────────────────────────────────────────

  const checkOllamaStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/models/llm`)
      if (response.ok) {
        const data = await response.json()
        setOllamaStatus({
          running: data.ollama.server_running,
          models: data.ollama.available_models
        })
        if (!data.ollama.server_running) {
          setLlmBackend('huggingface')
          setLlmModel('Qwen/Qwen2.5-0.5B-Instruct')
        }
      }
    } catch {
      setOllamaStatus({ running: false, models: [] })
      setLlmBackend('huggingface')
    }
  }

  const fetchEmbeddingModels = async () => {
    try {
      const response = await fetch(`${API_URL}/models/embedding`)
      if (response.ok) {
        setEmbeddingModels(await response.json())
      }
    } catch {
      console.error('임베딩 모델 정보 로드 실패')
    }
  }

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_URL}/rag/documents?collection=documents`)
      if (response.ok) {
        const data = await response.json()
        setDocuments(data.documents || [])
      }
    } catch {
      console.error('문서 목록 로드 실패')
    }
  }

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
      formData.append('overlap', '50')

      if (chunkMethod === 'semantic') {
        formData.append('semantic_threshold', semanticThreshold.toString())
      }
      if (chunkMethod === 'llm') {
        formData.append('llm_model', llmModel)
        formData.append('llm_backend', llmBackend)
      }

      const response = await fetch(`${API_URL}/rag/upload`, { method: 'POST', body: formData })

      if (response.ok) {
        const data = await response.json()
        setUploadStatus(`✅ ${data.filename} (${data.chunks_created}개 청크, 표 ${data.tables_found || 0}개)`)
        fetchDocuments()
      } else {
        const err = await response.json()
        setUploadStatus(`❌ 실패: ${err.detail || '알 수 없는 오류'}`)
      }
    } catch {
      setUploadStatus('❌ 업로드 실패')
    } finally {
      setLoading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleSearch = async () => {
    if (!ragQuery.trim()) return alert('질문을 입력하세요')

    setLoading(true)
    setRagResult(null)
    setGlobalAnswer('')
    setClarificationMessage(null)
    setClarificationOptions([])

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

      if (response.ok) {
        const data = await response.json()
        setRagResult({ query: ragQuery, results: data.results })
      }
    } catch {
      alert('검색 실패')
    } finally {
      setLoading(false)
    }
  }

  const handleAIAnswer = async (filterDoc?: string) => {
    if (!ragQuery.trim()) return

    setGlobalAnswerLoading(true)
    setGlobalAnswer('')
    setClarificationMessage(null)

    try {
      const endpoint = useAgent ? '/rag/agent' : '/rag/ask'
      const body = useAgent
        ? {
            query: ragQuery,
            collection: 'documents',
            n_results: 5,
            embedding_model: ragModel,
            llm_model: llmModel,
            llm_backend: llmBackend,
            agent_type: agentType,
            enable_clarification: enableClarification && !filterDoc,
            filter_doc: filterDoc || null
          }
        : {
            query: ragQuery,
            collection: 'documents',
            n_results: 5,
            embedding_model: ragModel,
            llm_model: llmModel,
            llm_backend: llmBackend,
            check_clarification: enableClarification && !filterDoc,
            filter_doc: filterDoc || null
          }

      const response = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })

      if (response.ok) {
        const data: RAGResponse = await response.json()

        if (data.needs_clarification && data.clarification_options) {
          setClarificationMessage(data.answer || '여러 문서가 검색되었습니다.')
          setClarificationOptions(data.clarification_options)
          setGlobalAnswer('')
        } else {
          setGlobalAnswer(data.answer || '답변 생성 실패')
          setClarificationMessage(null)
          setClarificationOptions([])
        }

        if (data.sources) {
          setRagResult({ query: ragQuery, results: data.sources })
        }
      }
    } catch {
      setGlobalAnswer('오류 발생')
    } finally {
      setGlobalAnswerLoading(false)
    }
  }

  const handleSelectDocument = (docName: string) => {
    setClarificationMessage(null)
    setClarificationOptions([])
    handleAIAnswer(docName)
  }

  const handleChunkAnswer = async (index: number, chunkText: string) => {
    if (!ragResult?.results) return

    const updated = [...ragResult.results]
    updated[index] = { ...updated[index], aiLoading: true }
    setRagResult({ ...ragResult, results: updated })

    try {
      const response = await fetch(`${API_URL}/rag/ask-chunk`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: ragQuery,
          chunk_text: chunkText,
          llm_model: llmModel,
          llm_backend: llmBackend
        })
      })
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

  // ─────────────────────────────────────────────────────────────
  // 메타데이터 표시 (개선됨)
  // ─────────────────────────────────────────────────────────────

  const renderMetadataDisplay = (result: SearchResult) => {
    const meta = result.metadata || {}
    const parts: string[] = []

    // 5개 필드만: sop_id, version, section, title
    if (meta.sop_id) parts.push(meta.sop_id)
    if (meta.version) parts.push(`v${meta.version}`)
    if (meta.section) parts.push(meta.section)
    if (meta.title) parts.push(meta.title)

    return parts.length > 0 ? parts.join(' > ') : null
  }

  // ─────────────────────────────────────────────────────────────
  // 렌더링
  // ─────────────────────────────────────────────────────────────

  return (
    <div className="app">
      <header className="header">
        <h1>📄 RAG System v6.0</h1>
        <p>Docling 기반 문서 파싱 | 에이전트 지원 | 표 파싱</p>
      </header>

      <main className="main">
        {/* Ollama 상태 */}
        <div className="status-bar">
          {ollamaStatus.running ? (
            <span className="status-ok">✅ Ollama ({ollamaStatus.models.length}개 모델)</span>
          ) : (
            <span className="status-warn">⚠️ Ollama 미실행 - HuggingFace 사용</span>
          )}
          <button onClick={checkOllamaStatus}>🔄</button>
        </div>

        {/* 모델 설정 */}
        <section className="settings-section">
          <h3>⚙️ 설정</h3>
          <div className="settings-grid">
            <div className="setting">
              <label>
                🔍 검색 모델
                <button className="info-btn" onClick={() => setShowModelInfo(!showModelInfo)}>ℹ️</button>
              </label>
              <select value={ragModel} onChange={(e) => setRagModel(e.target.value)}>
                {PRESET_MODELS.map(m => (
                  <option key={m.key} value={m.key}>
                    {m.compatible ? '' : '⚠️'} {m.name} ({m.dim}d)
                  </option>
                ))}
              </select>
            </div>

            <div className="setting">
              <label>🤖 LLM 백엔드</label>
              <select
                value={llmBackend}
                onChange={(e) => {
                  const backend = e.target.value as 'ollama' | 'huggingface'
                  setLlmBackend(backend)
                  setLlmModel(backend === 'ollama' ? 'qwen2.5:3b' : 'Qwen/Qwen2.5-0.5B-Instruct')
                }}
              >
                <option value="ollama" disabled={!ollamaStatus.running}>Ollama</option>
                <option value="huggingface">HuggingFace</option>
              </select>
            </div>

            <div className="setting">
              <label>💬 답변 모델</label>
              <select value={llmModel} onChange={(e) => setLlmModel(e.target.value)}>
                {llmBackend === 'ollama'
                  ? OLLAMA_MODELS.map(m => <option key={m.key} value={m.key}>{m.name} - {m.desc}</option>)
                  : HF_MODELS.map(m => <option key={m.key} value={m.key}>{m.name} - {m.desc}</option>)
                }
              </select>
            </div>
          </div>

          {/* 모델 정보 팝업 */}
          {showModelInfo && embeddingModels && (
            <div className="model-info-popup">
              <div className="popup-header">
                <h4>📊 임베딩 모델 (dim≤{embeddingModels.filter_criteria.max_dim})</h4>
                <button onClick={() => setShowModelInfo(false)}>×</button>
              </div>
              <div className="model-lists">
                <div>
                  <h5>✅ 호환 ({embeddingModels.compatible.length})</h5>
                  {embeddingModels.compatible.map(m => (
                    <div key={m.path} className="model-item">{m.name} ({m.dim}d)</div>
                  ))}
                </div>
                <div>
                  <h5>❌ 비호환 ({embeddingModels.incompatible.length})</h5>
                  {embeddingModels.incompatible.map(m => (
                    <div key={m.path} className="model-item warning">{m.name} ({m.dim}d)</div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </section>

        {/* 청킹 설정 */}
        <section className="chunk-section">
          <h3>📦 청킹 설정</h3>
          <div className="chunk-methods">
            {CHUNK_METHODS.map(m => (
              <button
                key={m.key}
                className={`chunk-btn ${chunkMethod === m.key ? 'active' : ''}`}
                onClick={() => setChunkMethod(m.key)}
                title={m.desc}
              >
                {m.name}
              </button>
            ))}
          </div>

          <div className="chunk-size-slider">
            <label>청크 크기: {chunkSize}자</label>
            <input
              type="range"
              min="100"
              max="1000"
              step="50"
              value={chunkSize}
              onChange={(e) => setChunkSize(Number(e.target.value))}
            />
          </div>

          {chunkMethod === 'semantic' && (
            <div className="chunk-size-slider">
              <label>Semantic 임계값: {semanticThreshold.toFixed(2)}</label>
              <input
                type="range"
                min="0.3"
                max="0.8"
                step="0.05"
                value={semanticThreshold}
                onChange={(e) => setSemanticThreshold(Number(e.target.value))}
              />
            </div>
          )}
        </section>

        {/* 에이전트 설정 */}
        <section className="agent-section">
          <h3>🤖 에이전트 설정</h3>
          <div className="agent-options">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={useAgent}
                onChange={(e) => setUseAgent(e.target.checked)}
              />
              에이전트 모드 사용
            </label>

            {useAgent && (
              <select value={agentType} onChange={(e) => setAgentType(e.target.value)}>
                {AGENT_TYPES.map(t => (
                  <option key={t.key} value={t.key}>{t.name} - {t.desc}</option>
                ))}
              </select>
            )}

            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={enableClarification}
                onChange={(e) => setEnableClarification(e.target.checked)}
              />
              🤔 되묻기 활성화
            </label>
          </div>
        </section>

        {/* 문서 업로드 */}
        <section className="upload-section">
          <h3>📁 문서 업로드</h3>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.doc,.txt,.md,.html,.xlsx,.pptx,.png,.jpg"
            onChange={handleFileUpload}
            disabled={loading}
          />
          {uploadStatus && <p className="upload-status">{uploadStatus}</p>}
        </section>

        {/* 문서 목록 */}
        {documents.length > 0 && (
          <section className="doc-list-section">
            <h3>📚 업로드된 문서 ({documents.length})</h3>
            <div className="doc-list">
              {documents.map((doc, i) => (
                <div key={i} className="doc-item">
                  <div className="doc-info">
                    <strong>{doc.doc_name}</strong>
                    <span className="doc-meta">
                      {doc.chunk_count}개 청크
                      {doc.chunk_method && ` • ${doc.chunk_method}`}
                    </span>
                  </div>
                  <button className="delete-btn" onClick={() => handleDeleteDocument(doc.doc_name)}>🗑️</button>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* 질문 입력 */}
        <section className="query-section">
          <h3>💬 질문</h3>
          <textarea
            value={ragQuery}
            onChange={(e) => setRagQuery(e.target.value)}
            placeholder="문서에 대해 질문하세요..."
            rows={3}
          />
          <div className="query-buttons">
            <button
              className="btn-search"
              onClick={handleSearch}
              disabled={loading || documents.length === 0}
            >
              🔍 검색만
            </button>
            <button
              className="btn-ai"
              onClick={async () => { await handleSearch(); await handleAIAnswer() }}
              disabled={loading || globalAnswerLoading || documents.length === 0}
            >
              ✨ {useAgent ? '에이전트' : 'AI'} 답변
            </button>
          </div>
        </section>

        {/* 되묻기 UI */}
        {clarificationMessage && clarificationOptions.length > 0 && (
          <section className="clarification-section">
            <div className="clarification-header">
              <span>🤔</span>
              <h3>확인이 필요합니다</h3>
            </div>
            <p className="clarification-message">{clarificationMessage}</p>

            <div className="clarification-options">
              {clarificationOptions.map((opt, i) => (
                <button
                  key={i}
                  className="option-btn"
                  onClick={() => handleSelectDocument(opt.doc_name)}
                  title={`관련도: ${(opt.score * 100).toFixed(0)}%`}
                >
                  <span className="option-icon">📄</span>
                  <span className="option-text">
                    {opt.display_text}
                    {opt.sections && opt.sections.length > 0 && (
                      <span className="option-sections">
                        {opt.sections.slice(0, 2).join(', ')}
                      </span>
                    )}
                  </span>
                  <span className="option-score">{(opt.score * 100).toFixed(0)}%</span>
                </button>
              ))}

              <button
                className="option-btn option-all"
                onClick={() => {
                  setClarificationMessage(null)
                  setClarificationOptions([])
                  setEnableClarification(false)
                  handleAIAnswer()
                }}
              >
                <span className="option-icon">📚</span>
                <span className="option-text">전체 문서 종합</span>
              </button>
            </div>
          </section>
        )}

        {/* AI 답변 */}
        {(globalAnswerLoading || globalAnswer) && !clarificationMessage && (
          <section className="answer-section">
            <h3>🤖 {useAgent ? '에이전트' : 'AI'} 답변</h3>
            {globalAnswerLoading ? (
              <div className="loading">
                <span className="spinner"></span>
                답변 생성 중... ({llmBackend})
              </div>
            ) : (
              <div className="answer-text">{globalAnswer}</div>
            )}
          </section>
        )}

        {/* 검색 결과 */}
        {ragResult?.results && ragResult.results.length > 0 && (
          <section className="results-section">
            <h3>📄 검색 결과 ({ragResult.results.length})</h3>

            {ragResult.results.map((r, idx) => (
              <div key={idx} className="result-card">
                <div className="result-header">
                  <div className="result-source">
                    <span className="source-doc">📄 {r.metadata?.doc_name || '문서'}</span>

                    {/* 개선된 메타데이터 표시 */}
                    {renderMetadataDisplay(r) && (
                      <span className="source-section">📌 {renderMetadataDisplay(r)}</span>
                    )}

                    <button
                      className="json-toggle"
                      onClick={() => {
                        const newSet = new Set(expandedMeta)
                        expandedMeta.has(idx) ? newSet.delete(idx) : newSet.add(idx)
                        setExpandedMeta(newSet)
                      }}
                    >
                      {expandedMeta.has(idx) ? '▼' : '▶'} JSON
                    </button>
                  </div>

                  <div className="result-scores">
                    <span
                      className="confidence-badge"
                      style={{ backgroundColor: getConfidenceColor(r.confidence) }}
                    >
                      {r.confidence_text || r.confidence || 'medium'}
                    </span>
                    <span
                      className="similarity-score"
                      style={{ color: getSimilarityColor(r.similarity) }}
                    >
                      {(r.similarity * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                {/* JSON 메타데이터 */}
                {expandedMeta.has(idx) && (
                  <pre className="metadata-json">
                    {JSON.stringify(r.metadata, null, 2)}
                  </pre>
                )}

                <div className="result-text">{r.text}</div>

                <button
                  className="chunk-answer-btn"
                  onClick={() => handleChunkAnswer(idx, r.text)}
                  disabled={r.aiLoading}
                >
                  {r.aiLoading ? '생성 중...' : '🤖 이 청크로 답변'}
                </button>

                {r.aiAnswer && (
                  <div className="chunk-answer">
                    <div className="chunk-answer-title">💡 청크 기반 답변</div>
                    <div className="chunk-answer-text">{r.aiAnswer}</div>
                  </div>
                )}
              </div>
            ))}
          </section>
        )}

        {ragResult && (!ragResult.results || ragResult.results.length === 0) && !loading && (
          <div className="no-results">관련 문서를 찾을 수 없습니다.</div>
        )}
      </main>

      <footer className="footer">
        v6.0 - Docling 파싱 | 에이전트 | 표 지원 | 제N조 메타데이터
      </footer>
    </div>
  )
}

export default App