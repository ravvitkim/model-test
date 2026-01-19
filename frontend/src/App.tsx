import { useState, useRef, useEffect } from 'react'
import './App.css'

// ═══════════════════════════════════════════════════════════════════════════
// 타입 정의
// ═══════════════════════════════════════════════════════════════════════════

interface CompareResult {
  similarity: number
  interpretation: string
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

interface SearchResult {
  text: string
  similarity: number
  metadata: {
    doc_name: string
    doc_title?: string
    chunk_index: number
    total_chunks?: number
    chunk_method?: string
    article_num?: string
    article_type?: string
    section?: string
  }
  aiAnswer?: string
  aiLoading?: boolean
}

interface ClarificationOption {
  doc_name: string
  display_text: string
  score: number
}

interface RAGResponse {
  query: string
  answer?: string
  results?: SearchResult[]
  sources?: SearchResult[]
  needs_clarification?: boolean
  clarification_options?: ClarificationOption[]
}

interface DocumentInfo {
  doc_name: string
  doc_title?: string
  chunk_count: number
  chunk_method?: string
}

interface LLMModelsResponse {
  ollama: {
    server_running: boolean
    available_models: string[]
    models: Array<{ key: string; name: string; desc: string; vram: string; available: boolean }>
  }
  huggingface: {
    models: Array<{ key: string; name: string; desc: string }>
  }
}

// 임베딩 모델 스펙 타입 ← NEW
interface EmbeddingModelSpec {
  path: string
  name: string
  dim: number
  memory_mb: number
  lang: string
  desc: string
  compatible: boolean
  warning?: boolean
}

interface EmbeddingModelsResponse {
  all: EmbeddingModelSpec[]
  compatible: EmbeddingModelSpec[]
  incompatible: EmbeddingModelSpec[]
  filter_criteria: {
    max_dim: number
    max_memory_mb: number
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 프리셋 모델 (호환성 정보 포함) ← UPDATED
// ═══════════════════════════════════════════════════════════════════════════

const PRESET_MODELS = [
  { key: 'ko-sroberta', name: 'Ko-SROBERTA', desc: '한국어 추천', dim: 768, mem: 440, compatible: true },
  { key: 'ko-sbert', name: 'Ko-SBERT', desc: '한국어', dim: 768, mem: 440, compatible: true },
  { key: 'ko-simcse', name: 'Ko-SimCSE', desc: '한국어', dim: 768, mem: 440, compatible: true },
  { key: 'qwen3-0.6b', name: 'Qwen3-0.6B', desc: '다국어 경량', dim: 1024, mem: 600, compatible: true },
  { key: 'qwen3-4b', name: 'Qwen3-4B', desc: '⚠️ dim/mem 초과', dim: 2560, mem: 4000, compatible: false },
  { key: 'multilingual-minilm', name: 'MiniLM 다국어', desc: '경량', dim: 384, mem: 470, compatible: true },
  { key: 'multilingual-e5', name: 'E5 다국어', desc: '고성능', dim: 1024, mem: 1200, compatible: true },
  { key: 'bge-m3', name: 'BGE-M3', desc: '최신', dim: 1024, mem: 1300, compatible: true },
  { key: 'minilm', name: 'MiniLM', desc: '영어 경량', dim: 384, mem: 90, compatible: true },
  { key: 'mpnet', name: 'MPNet', desc: '영어 고성능', dim: 768, mem: 420, compatible: true },
]

const OLLAMA_MODELS = [
  { key: 'qwen2.5:0.5b', name: 'Qwen2.5-0.5B', desc: '초경량 (1GB)' },
  { key: 'qwen2.5:1.5b', name: 'Qwen2.5-1.5B', desc: '경량 (2GB)' },
  { key: 'qwen2.5:3b', name: 'Qwen2.5-3B', desc: '추천 (3GB)' },
  { key: 'qwen2.5:7b', name: 'Qwen2.5-7B', desc: '고성능 (5GB)' },
  { key: 'qwen3:4b', name: 'Qwen3-4B', desc: '최신 추천 (4GB)' },
  { key: 'llama3.2:3b', name: 'Llama3.2-3B', desc: '경량 (3GB)' },
  { key: 'gemma2:2b', name: 'Gemma2-2B', desc: '경량 (2GB)' },
  { key: 'mistral:7b', name: 'Mistral-7B', desc: '영어 특화 (5GB)' },
]

const HF_MODELS = [
  { key: 'Qwen/Qwen2.5-0.5B-Instruct', name: 'Qwen2.5-0.5B', desc: '초경량' },
  { key: 'Qwen/Qwen2.5-1.5B-Instruct', name: 'Qwen2.5-1.5B', desc: '경량' },
  { key: 'Qwen/Qwen2.5-3B-Instruct', name: 'Qwen2.5-3B', desc: 'VRAM 6GB+' },
  { key: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', name: 'TinyLlama', desc: '영어 특화' },
]

// 청킹 방식 정의 ← NEW
const CHUNK_METHODS = [
  { key: 'article', name: '📜 조항 단위', desc: 'SOP/법률 권장', icon: '📜' },
  { key: 'recursive', name: '🔄 Recursive', desc: '랭체인 스타일', icon: '🔄' },
  { key: 'semantic', name: '🧠 Semantic', desc: '의미 기반 (느림)', icon: '🧠' },
  { key: 'sentence', name: '📝 문장 단위', desc: '빠름', icon: '📝' },
  { key: 'paragraph', name: '📄 문단 단위', desc: '중간', icon: '📄' },
  { key: 'llm', name: '🤖 LLM 파싱', desc: '가장 정교 (가장 느림)', icon: '🤖' },
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
  // 텍스트 비교
  const [text1, setText1] = useState('')
  const [text2, setText2] = useState('')
  const [selectedModel, setSelectedModel] = useState('ko-sroberta')
  const [result, setResult] = useState<CompareResult | null>(null)
  const [multiResult, setMultiResult] = useState<MultiModelResult | null>(null)
  const [selectedModels, setSelectedModels] = useState<string[]>(['ko-sroberta', 'qwen3-0.6b'])
  const [texts, setTexts] = useState<string[]>(['', '', ''])
  const [matrixResult, setMatrixResult] = useState<MatrixResult | null>(null)

  // RAG
  const [ragQuery, setRagQuery] = useState('')
  const [ragResult, setRagResult] = useState<RAGResponse | null>(null)
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [uploadStatus, setUploadStatus] = useState<string>('')
  const [ragModel, setRagModel] = useState('ko-sroberta')
  
  // LLM
  const [llmBackend, setLlmBackend] = useState<'ollama' | 'huggingface'>('ollama')
  const [llmModel, setLlmModel] = useState('qwen2.5:3b')
  const [ollamaStatus, setOllamaStatus] = useState<{ running: boolean; models: string[] }>({ running: false, models: [] })
  
  // 청킹 ← UPDATED
  const [chunkMethod, setChunkMethod] = useState<string>('article')
  const [chunkSize, setChunkSize] = useState<number>(500)
  const [semanticThreshold, setSemanticThreshold] = useState<number>(0.5)  // NEW
  const [chunkLlmModel, setChunkLlmModel] = useState<string>('qwen2.5:3b')  // NEW (LLM 파싱용)
  
  // 되묻기
  const [enableClarification, setEnableClarification] = useState(true)
  const [clarificationMessage, setClarificationMessage] = useState<string | null>(null)
  const [clarificationOptions, setClarificationOptions] = useState<ClarificationOption[]>([])
  
  // 임베딩 모델 정보 ← NEW
  const [showModelInfo, setShowModelInfo] = useState(false)
  const [embeddingModels, setEmbeddingModels] = useState<EmbeddingModelsResponse | null>(null)
  
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [globalAnswer, setGlobalAnswer] = useState<string>('')
  const [globalAnswerLoading, setGlobalAnswerLoading] = useState(false)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'single' | 'multi' | 'matrix' | 'rag'>('rag')

  useEffect(() => {
    checkOllamaStatus()
    fetchEmbeddingModels()
  }, [])

  const checkOllamaStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/models/llm`)
      if (response.ok) {
        const data: LLMModelsResponse = await response.json()
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

  // 임베딩 모델 정보 가져오기 ← NEW
  const fetchEmbeddingModels = async () => {
    try {
      const response = await fetch(`${API_URL}/models/embedding`)
      if (response.ok) {
        const data: EmbeddingModelsResponse = await response.json()
        setEmbeddingModels(data)
      }
    } catch {
      console.error('임베딩 모델 정보 로드 실패')
    }
  }

  const handleCompare = async () => {
    if (!text1.trim() || !text2.trim()) return alert('두 텍스트를 모두 입력해주세요.')
    
    // 모델 호환성 체크
    const model = PRESET_MODELS.find(m => m.key === selectedModel)
    if (model && !model.compatible) {
      if (!confirm(`⚠️ ${model.name}은 dim=${model.dim}, mem=${model.mem}MB로 권장 범위(dim≤1024, mem≤1300MB)를 초과합니다. 계속하시겠습니까?`)) {
        return
      }
    }
    
    setLoading(true)
    setResult(null)
    try {
      const response = await fetch(`${API_URL}/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text1, text2, model: selectedModel })
      })
      if (response.ok) setResult(await response.json())
    } catch { alert('서버 연결 실패') }
    finally { setLoading(false) }
  }

  const handleMultiCompare = async () => {
    if (!text1.trim() || !text2.trim()) return alert('두 텍스트를 모두 입력해주세요.')
    if (selectedModels.length < 1) return alert('최소 1개 모델을 선택해주세요.')
    setLoading(true)
    setMultiResult(null)
    try {
      const response = await fetch(`${API_URL}/compare/models`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text1, text2, models: selectedModels })
      })
      if (response.ok) setMultiResult(await response.json())
    } catch { alert('서버 연결 실패') }
    finally { setLoading(false) }
  }

  const handleMatrixCompare = async () => {
    const validTexts = texts.filter(t => t.trim())
    if (validTexts.length < 2) return alert('최소 2개 텍스트를 입력해주세요.')
    setLoading(true)
    setMatrixResult(null)
    try {
      const response = await fetch(`${API_URL}/compare/matrix`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts: validTexts, model: selectedModel })
      })
      if (response.ok) setMatrixResult(await response.json())
    } catch { alert('서버 연결 실패') }
    finally { setLoading(false) }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    
    // 모델 호환성 체크
    const model = PRESET_MODELS.find(m => m.key === ragModel)
    if (model && !model.compatible) {
      if (!confirm(`⚠️ ${model.name}은 권장 범위를 초과합니다. 계속하시겠습니까?`)) {
        return
      }
    }
    
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
      
      // Semantic 분할용 threshold
      if (chunkMethod === 'semantic') {
        formData.append('semantic_threshold', semanticThreshold.toString())
      }
      
      // LLM 파싱용 모델 설정
      if (chunkMethod === 'llm') {
        formData.append('llm_model', chunkLlmModel)
        formData.append('llm_backend', llmBackend)
      }
      
      const response = await fetch(`${API_URL}/rag/upload`, { method: 'POST', body: formData })
      if (response.ok) {
        const data = await response.json()
        setUploadStatus(`✅ ${data.filename} (${data.chunks_created}개 조각, ${data.chunk_method})`)
        fetchDocuments()
      } else {
        const errorData = await response.json()
        setUploadStatus(`❌ 업로드 실패: ${errorData.detail || '알 수 없는 오류'}`)
      }
    } catch { setUploadStatus('❌ 업로드 실패') }
    finally { 
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
    } catch { console.error('문서 목록 로드 실패') }
  }

  const handleRAGSearch = async () => {
    if (!ragQuery.trim()) return alert('질문을 입력해주세요.')
    setLoading(true)
    setRagResult(null)
    setGlobalAnswer('')
    setClarificationMessage(null)
    setClarificationOptions([])
    try {
      const response = await fetch(`${API_URL}/rag/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: ragQuery, collection: 'documents', n_results: 5, model: ragModel })
      })
      if (response.ok) setRagResult(await response.json())
    } catch { alert('검색 실패') }
    finally { setLoading(false) }
  }

  const handleAIAnswer = async (filterDoc?: string) => {
    if (!ragQuery.trim()) return
    setGlobalAnswerLoading(true)
    setGlobalAnswer('')
    setClarificationMessage(null)
    
    try {
      const response = await fetch(`${API_URL}/rag/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: ragQuery,
          collection: 'documents',
          n_results: 5,
          embedding_model: ragModel,
          llm_model: llmModel,
          llm_backend: llmBackend,
          check_clarification: enableClarification && !filterDoc,
          filter_doc: filterDoc || null
        })
      })
      if (response.ok) {
        const data: RAGResponse = await response.json()
        
        if (data.needs_clarification && data.clarification_options) {
          setClarificationMessage(data.answer || '')
          setClarificationOptions(data.clarification_options)
          setGlobalAnswer('')
        } else {
          setGlobalAnswer(data.answer || '답변 생성 실패')
          setClarificationMessage(null)
          setClarificationOptions([])
        }
        
        if (data.sources) {
          setRagResult(prev => prev ? { ...prev, results: data.sources } : { query: ragQuery, results: data.sources })
        }
      }
    } catch { setGlobalAnswer('오류 발생') }
    finally { setGlobalAnswerLoading(false) }
  }

  const handleSelectDocument = (docName: string) => {
    setClarificationMessage(null)
    setClarificationOptions([])
    handleAIAnswer(docName)
  }

  const handleChunkAIAnswer = async (index: number, chunkText: string) => {
    if (!ragResult?.results) return
    const updatedResults = [...ragResult.results]
    updatedResults[index] = { ...updatedResults[index], aiLoading: true }
    setRagResult({ ...ragResult, results: updatedResults })

    try {
      const response = await fetch(`${API_URL}/rag/ask-chunk`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: ragQuery, chunk_text: chunkText, llm_model: llmModel, llm_backend: llmBackend })
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
    } catch { alert('삭제 실패') }
  }

  const handleTabChange = (tab: 'single' | 'multi' | 'matrix' | 'rag') => {
    setActiveTab(tab)
    if (tab === 'rag') fetchDocuments()
  }

  const getArticleInfo = (metadata: SearchResult['metadata']) => {
    const parts = []
    if (metadata.article_type === 'article' && metadata.article_num) parts.push(`제${metadata.article_num}조`)
    else if (metadata.article_type === 'chapter' && metadata.article_num) parts.push(`제${metadata.article_num}장`)
    else if (metadata.article_num) parts.push(`${metadata.article_num}`)
    if (metadata.section) parts.push(metadata.section)
    return parts.join(' / ')
  }

  // 모델 선택 렌더링 (호환성 표시 포함) ← NEW
  const renderModelSelect = (value: string, onChange: (v: string) => void, showWarning: boolean = true) => (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {PRESET_MODELS.map(m => (
        <option key={m.key} value={m.key} style={{ color: m.compatible ? 'inherit' : '#f97316' }}>
          {m.compatible ? '' : '⚠️ '}{m.name} {showWarning && `(${m.dim}d, ${m.mem}MB)`}
        </option>
      ))}
    </select>
  )

  return (
    <div className="app">
      <header className="header">
        <h1 className="title">🔍 텍스트 유사도 + RAG v5.0</h1>
        <p className="subtitle">확장 청킹 (Recursive/Semantic/LLM) + 모델 필터링 + Ollama 지원</p>
      </header>

      <div className="tabs">
        {(['single', 'multi', 'matrix', 'rag'] as const).map(tab => (
          <button key={tab} className={`tab ${activeTab === tab ? 'active' : ''}`} onClick={() => handleTabChange(tab)}>
            {tab === 'single' && '단일 비교'}
            {tab === 'multi' && '모델 비교'}
            {tab === 'matrix' && '매트릭스'}
            {tab === 'rag' && '📄 RAG'}
          </button>
        ))}
      </div>

      <main className="main">
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
              <label>모델 <button className="info-btn" onClick={() => setShowModelInfo(!showModelInfo)}>ℹ️</button></label>
              {renderModelSelect(selectedModel, setSelectedModel)}
            </div>
            
            {/* 모델 정보 팝업 */}
            {showModelInfo && embeddingModels && (
              <div className="model-info-popup">
                <div className="popup-header">
                  <h4>📊 임베딩 모델 필터 (dim≤{embeddingModels.filter_criteria.max_dim}, mem≤{embeddingModels.filter_criteria.max_memory_mb}MB)</h4>
                  <button onClick={() => setShowModelInfo(false)}>×</button>
                </div>
                <div className="model-lists">
                  <div className="compatible-list">
                    <h5>✅ 호환 ({embeddingModels.compatible.length})</h5>
                    {embeddingModels.compatible.map(m => (
                      <div key={m.path} className="model-item">
                        <span>{m.name}</span>
                        <span className="model-spec">{m.dim}d / {m.memory_mb}MB</span>
                      </div>
                    ))}
                  </div>
                  <div className="incompatible-list">
                    <h5>❌ 비호환 ({embeddingModels.incompatible.length})</h5>
                    {embeddingModels.incompatible.map(m => (
                      <div key={m.path} className="model-item warning">
                        <span>{m.name}</span>
                        <span className="model-spec">{m.dim}d / {m.memory_mb}MB</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
            
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
                <label key={m.key} className={`model-chip ${selectedModels.includes(m.key) ? 'selected' : ''} ${!m.compatible ? 'incompatible' : ''}`} title={`${m.dim}d, ${m.mem}MB${!m.compatible ? ' (⚠️ 권장 초과)' : ''}`}>
                  <input type="checkbox" checked={selectedModels.includes(m.key)} onChange={() => setSelectedModels(prev => prev.includes(m.key) ? prev.filter(k => k !== m.key) : [...prev, m.key])} />
                  {!m.compatible && '⚠️ '}{m.name}
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
                    <span className="result-score" style={{ color: getSimilarityColor(r.similarity) }}>{(r.similarity * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {activeTab === 'matrix' && (
          <>
            <div className="matrix-inputs">
              {texts.map((text, i) => (
                <div key={i} className="matrix-row">
                  <span className="row-num">{i + 1}</span>
                  <textarea value={text} onChange={(e) => { const newTexts = [...texts]; newTexts[i] = e.target.value; setTexts(newTexts) }} placeholder={`텍스트 ${i + 1}`} rows={2} />
                  {texts.length > 2 && <button className="remove-btn" onClick={() => setTexts(texts.filter((_, j) => j !== i))}>×</button>}
                </div>
              ))}
              {texts.length < 10 && <button className="add-btn" onClick={() => setTexts([...texts, ''])}>+ 추가</button>}
            </div>
            <button className="primary-btn" onClick={handleMatrixCompare} disabled={loading}>{loading ? '계산 중...' : '매트릭스 생성'}</button>
            {matrixResult && (
              <div className="matrix-table-wrap">
                <table className="matrix-table">
                  <thead><tr><th></th>{matrixResult.texts.map((_, i) => <th key={i}>{i + 1}</th>)}</tr></thead>
                  <tbody>
                    {matrixResult.similarity_matrix.map((row, i) => (
                      <tr key={i}>
                        <td className="row-head">{i + 1}</td>
                        {row.map((score, j) => <td key={j} style={{ backgroundColor: i === j ? '#333' : `${getSimilarityColor(score)}33`, color: getSimilarityColor(score) }}>{(score * 100).toFixed(0)}%</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}

        {activeTab === 'rag' && (
          <>
            <div className="ollama-status">
              {ollamaStatus.running ? (
                <span className="status-ok">✅ Ollama 실행 중 ({ollamaStatus.models.length}개 모델)</span>
              ) : (
                <span className="status-warn">⚠️ Ollama 미실행 - HuggingFace 사용</span>
              )}
              <button className="refresh-btn" onClick={checkOllamaStatus}>🔄</button>
            </div>

            <div className="settings-row">
              <div className="setting">
                <label>🔍 검색 모델</label>
                {renderModelSelect(ragModel, setRagModel, false)}
              </div>
              <div className="setting">
                <label>🤖 LLM 백엔드</label>
                <select value={llmBackend} onChange={(e) => {
                  const backend = e.target.value as 'ollama' | 'huggingface'
                  setLlmBackend(backend)
                  setLlmModel(backend === 'ollama' ? 'qwen2.5:3b' : 'Qwen/Qwen2.5-0.5B-Instruct')
                }}>
                  <option value="ollama" disabled={!ollamaStatus.running}>Ollama (로컬)</option>
                  <option value="huggingface">HuggingFace</option>
                </select>
              </div>
              <div className="setting">
                <label>💬 답변 모델</label>
                <select value={llmModel} onChange={(e) => setLlmModel(e.target.value)}>
                  {llmBackend === 'ollama' ? OLLAMA_MODELS.map(m => <option key={m.key} value={m.key}>{m.name} - {m.desc}</option>) : HF_MODELS.map(m => <option key={m.key} value={m.key}>{m.name} - {m.desc}</option>)}
                </select>
              </div>
            </div>

            {/* 청킹 설정 - 확장됨 ← UPDATED */}
            <div className="chunk-settings">
              <label className="chunk-label">📦 청킹 방식</label>
              <div className="chunk-method-grid">
                {CHUNK_METHODS.map(m => (
                  <button 
                    key={m.key} 
                    className={`chunk-btn ${chunkMethod === m.key ? 'active' : ''}`} 
                    onClick={() => setChunkMethod(m.key)}
                    title={m.desc}
                  >
                    <span className="chunk-icon">{m.icon}</span>
                    <span className="chunk-name">{m.name.replace(/^[^\s]+\s/, '')}</span>
                  </button>
                ))}
              </div>
              
              <div className="chunk-size">
                <span>최대 조각 크기: {chunkSize}자</span>
                <input type="range" min="200" max="2000" step="100" value={chunkSize} onChange={(e) => setChunkSize(Number(e.target.value))} />
              </div>
              
              {/* Semantic 분할 옵션 ← NEW */}
              {chunkMethod === 'semantic' && (
                <div className="semantic-options">
                  <span>🧠 유사도 임계값: {semanticThreshold.toFixed(2)}</span>
                  <input 
                    type="range" 
                    min="0.3" 
                    max="0.8" 
                    step="0.05" 
                    value={semanticThreshold} 
                    onChange={(e) => setSemanticThreshold(Number(e.target.value))} 
                  />
                  <span className="hint">낮을수록 더 작게 분할</span>
                </div>
              )}
              
              {/* LLM 파싱 옵션 ← NEW */}
              {chunkMethod === 'llm' && (
                <div className="llm-chunk-options">
                  <span>🤖 파싱용 LLM:</span>
                  <select value={chunkLlmModel} onChange={(e) => setChunkLlmModel(e.target.value)}>
                    {llmBackend === 'ollama' 
                      ? OLLAMA_MODELS.map(m => <option key={m.key} value={m.key}>{m.name}</option>) 
                      : HF_MODELS.map(m => <option key={m.key} value={m.key}>{m.name}</option>)
                    }
                  </select>
                  <span className="hint">⚠️ 가장 느리지만 가장 정확</span>
                </div>
              )}
            </div>

            <div className="clarification-toggle">
              <label>
                <input type="checkbox" checked={enableClarification} onChange={(e) => setEnableClarification(e.target.checked)} />
                🤔 여러 문서에서 결과 시 되묻기 (에이전트 모드)
              </label>
            </div>

            <div className="upload-section">
              <label>📁 문서 업로드 (PDF, DOCX, TXT)</label>
              <input ref={fileInputRef} type="file" accept=".pdf,.docx,.doc,.txt,.md,.html" onChange={handleFileUpload} disabled={loading} />
              {uploadStatus && <p className="status">{uploadStatus}</p>}
            </div>

            {documents.length > 0 && (
              <div className="doc-list">
                <label>📚 업로드된 문서</label>
                {documents.map((doc, i) => (
                  <div key={i} className="doc-item">
                    <div>
                      <strong>{doc.doc_name}</strong>
                      <span className="doc-meta">{doc.chunk_count}개 조각{doc.chunk_method && ` • ${doc.chunk_method}`}</span>
                    </div>
                    <button onClick={() => handleDeleteDocument(doc.doc_name)}>🗑️</button>
                  </div>
                ))}
              </div>
            )}

            <div className="query-section">
              <label>💬 질문</label>
              <textarea value={ragQuery} onChange={(e) => setRagQuery(e.target.value)} placeholder="문서에 대해 질문하세요... (예: 손 씻는 방법을 알려주세요)" rows={3} />
              <div className="query-btns">
                <button className="search-btn" onClick={handleRAGSearch} disabled={loading || documents.length === 0}>🔍 검색만</button>
                <button className="ai-btn" onClick={async () => { await handleRAGSearch(); await handleAIAnswer() }} disabled={loading || globalAnswerLoading || documents.length === 0}>✨ 검색 + AI 답변</button>
              </div>
            </div>

            {clarificationMessage && clarificationOptions.length > 0 && (
              <div className="clarification-box">
                <div className="clarification-header">
                  <span className="agent-icon">🤔</span>
                  <h3>확인이 필요합니다</h3>
                </div>
                <p className="clarification-msg">{clarificationMessage}</p>
                
                <div className="clarification-options">
                  {clarificationOptions.map((option, i) => (
                    <button 
                      key={i} 
                      className="option-btn" 
                      onClick={() => handleSelectDocument(option.doc_name)}
                      title={`정확도: ${(option.score * 100).toFixed(1)}%`}
                    >
                      <span className="doc-icon">📄</span>
                      <span className="option-text">{option.display_text}</span>
                      <span className="option-score">{(option.score * 100).toFixed(0)}%</span>
                    </button>
                  ))}
                  
                  <button 
                    className="option-btn all" 
                    onClick={() => { 
                      setClarificationMessage(null); 
                      setClarificationOptions([]); 
                      setEnableClarification(false); 
                      handleAIAnswer(); 
                    }}
                  >
                    <span className="doc-icon">📚</span>
                    <span className="option-text">전체 문서 내용 요약하기</span>
                  </button>
                </div>
              </div>
            )}

            {(globalAnswerLoading || globalAnswer) && !clarificationMessage && (
              <div className="global-answer">
                <h3>🤖 AI 종합 답변</h3>
                {globalAnswerLoading ? <div className="loading-answer"><span className="spinner"></span>답변 생성 중... ({llmBackend === 'ollama' ? 'Ollama' : 'HuggingFace'})</div> : <div className="answer-text">{globalAnswer}</div>}
              </div>
            )}

            {ragResult?.results && ragResult.results.length > 0 && (
              <div className="search-results">
                <h3>📄 관련 문서 조각 ({ragResult.results.length}개)</h3>
                {ragResult.results.map((r, idx) => (
                  <div key={idx} className="result-card">
                    <div className="card-header">
                      <div className="source-info">
                        <span className="source-file">📄 {r.metadata?.doc_name}</span>
                        {getArticleInfo(r.metadata) && <span className="article-info">📌 {getArticleInfo(r.metadata)}</span>}
                      </div>
                      <div className="relevance" style={{ color: getSimilarityColor(r.similarity) }}>
                        <span className="relevance-value">{getSimilarityLabel(r.similarity)}</span>
                        <span className="relevance-percent">{(r.similarity * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                    <div className="card-content">{r.text}</div>
                    <button className="chunk-ai-btn" onClick={() => handleChunkAIAnswer(idx, r.text)} disabled={r.aiLoading}>{r.aiLoading ? '생성 중...' : '🤖 이 내용으로 답변 생성'}</button>
                    {r.aiAnswer && <div className="chunk-answer"><div className="chunk-answer-title">💡 AI 답변</div><div className="chunk-answer-text">{r.aiAnswer}</div></div>}
                  </div>
                ))}
              </div>
            )}

            {ragResult && (!ragResult.results || ragResult.results.length === 0) && !loading && <div className="no-results">관련 문서를 찾을 수 없습니다.</div>}
          </>
        )}
      </main>

      <footer className="footer">v5.0 - 확장 청킹 (Recursive/Semantic/LLM) + 임베딩 모델 필터링 (dim≤1024, mem≤1300MB)</footer>
    </div>
  )
}

export default App