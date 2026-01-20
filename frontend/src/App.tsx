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

// 메타데이터 타입 (확장됨) ← UPDATED
interface ChunkMetadata {
  doc_name: string
  doc_title?: string
  chunk_index: number
  total_chunks?: number
  chunk_method?: string
  article_num?: string
  article_type?: string
  section?: string
  char_count?: number
  model?: string
  block_type?: string
  page?: number
  chunk_part?: string
}

// 검색 결과 타입 (confidence 추가) ← UPDATED
interface SearchResult {
  text: string
  similarity: number
  metadata: ChunkMetadata
  id?: string
  confidence?: 'high' | 'medium' | 'low'
  confidence_text?: string
  interpretation?: string
  aiAnswer?: string
  aiLoading?: boolean
}

// 품질 요약 타입 ← NEW
interface QualitySummary {
  avg_similarity?: number
  max_similarity?: number
  min_similarity?: number
  high_confidence_count?: number
  threshold_used?: number
  message?: string
}

interface ClarificationOption {
  doc_name: string
  display_text: string
  score: number
}

// RAG 응답 타입 (품질 요약 추가) ← UPDATED
interface RAGResponse {
  query: string
  answer?: string
  results?: SearchResult[]
  sources?: SearchResult[]
  needs_clarification?: boolean
  clarification_options?: ClarificationOption[]
  quality_summary?: QualitySummary
  quality_warning?: string
}

interface DocumentInfo {
  doc_name: string
  doc_title?: string
  chunk_count: number
  chunk_method?: string
  model?: string
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
// 프리셋 모델
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
  if (score >= 0.65) return '#22c55e'  // high
  if (score >= 0.35) return '#eab308'  // medium
  return '#ef4444'                      // low
}

const getSimilarityLabel = (score: number) => {
  if (score >= 0.85) return '매우 높음'
  if (score >= 0.65) return '높음'
  if (score >= 0.50) return '보통'
  if (score >= 0.35) return '낮음'
  return '매우 낮음'
}

// 신뢰도 컬러 및 라벨 ← NEW
const getConfidenceInfo = (confidence?: string) => {
  switch (confidence) {
    case 'high':
      return { color: '#22c55e', emoji: '🟢', label: '신뢰도 높음' }
    case 'medium':
      return { color: '#eab308', emoji: '🟡', label: '참고용' }
    case 'low':
      return { color: '#ef4444', emoji: '🔴', label: '관련성 낮음' }
    default:
      return { color: '#666', emoji: '⚪', label: '알 수 없음' }
  }
}

// 조항 타입 한글 변환 ← NEW
const getArticleTypeLabel = (type?: string) => {
  switch (type) {
    case 'article': return '조'
    case 'chapter': return '장'
    case 'section': return '절'
    case 'subsection': return '항'
    case 'item': return '호'
    case 'subitem': return '목'
    case 'intro': return '서문'
    case 'page': return '페이지'
    default: return type || ''
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 메타데이터 시각화 컴포넌트 ← NEW
// ═══════════════════════════════════════════════════════════════════════════

interface MetadataViewerProps {
  metadata: ChunkMetadata
  isExpanded?: boolean
  onToggle?: () => void
}

const MetadataViewer = ({ metadata, isExpanded = false, onToggle }: MetadataViewerProps) => {
  const [expanded, setExpanded] = useState(isExpanded)

  const handleToggle = () => {
    setExpanded(!expanded)
    onToggle?.()
  }

  // 메타데이터를 카테고리별로 분류
  const categories = {
    document: {
      label: '📄 문서 정보',
      items: [
        { key: 'doc_name', label: '파일명', value: metadata.doc_name },
        { key: 'doc_title', label: '제목', value: metadata.doc_title },
        { key: 'model', label: '임베딩 모델', value: metadata.model },
      ].filter(item => item.value)
    },
    structure: {
      label: '📌 구조 정보',
      items: [
        { key: 'article_type', label: '유형', value: metadata.article_type ? getArticleTypeLabel(metadata.article_type) : undefined },
        { key: 'article_num', label: '번호', value: metadata.article_num },
        { key: 'section', label: '섹션', value: metadata.section },
        { key: 'block_type', label: '블록 타입', value: metadata.block_type },
        { key: 'page', label: '페이지', value: metadata.page },
      ].filter(item => item.value !== undefined)
    },
    chunk: {
      label: '🧩 청크 정보',
      items: [
        { key: 'chunk_index', label: '청크 번호', value: `${metadata.chunk_index + 1}` },
        { key: 'total_chunks', label: '전체 청크', value: metadata.total_chunks },
        { key: 'chunk_method', label: '청킹 방식', value: metadata.chunk_method },
        { key: 'chunk_part', label: '분할', value: metadata.chunk_part },
        { key: 'char_count', label: '문자 수', value: metadata.char_count ? `${metadata.char_count}자` : undefined },
      ].filter(item => item.value !== undefined)
    }
  }

  return (
    <div className="metadata-viewer">
      <button className="metadata-toggle" onClick={handleToggle}>
        <span>{expanded ? '📋' : '📋'} 메타데이터</span>
        <span className="toggle-icon">{expanded ? '▼' : '▶'}</span>
      </button>
      
      {expanded && (
        <div className="metadata-content">
          {Object.entries(categories).map(([key, category]) => (
            category.items.length > 0 && (
              <div key={key} className="metadata-category">
                <div className="category-label">{category.label}</div>
                <div className="metadata-items">
                  {category.items.map(item => (
                    <div key={item.key} className="metadata-item">
                      <span className="item-key">{item.label}</span>
                      <span className="item-value">{String(item.value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )
          ))}
          
          {/* Raw JSON 토글 */}
          <details className="raw-json">
            <summary>🔧 Raw JSON</summary>
            <pre>{JSON.stringify(metadata, null, 2)}</pre>
          </details>
        </div>
      )}
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════
// 품질 요약 컴포넌트 ← NEW
// ═══════════════════════════════════════════════════════════════════════════

interface QualitySummaryCardProps {
  summary: QualitySummary
  warning?: string
  resultCount: number
}

const QualitySummaryCard = ({ summary, warning, resultCount }: QualitySummaryCardProps) => {
  if (summary.message) {
    return <div className="quality-summary empty">{summary.message}</div>
  }

  const highCount = summary.high_confidence_count || 0
  const highPercent = resultCount > 0 ? Math.round((highCount / resultCount) * 100) : 0

  return (
    <div className="quality-summary">
      <div className="quality-header">
        <span className="quality-title">📊 검색 품질</span>
        {warning && <span className="quality-warning">⚠️</span>}
      </div>
      
      <div className="quality-metrics">
        <div className="metric">
          <span className="metric-value" style={{ color: getSimilarityColor(summary.avg_similarity || 0) }}>
            {((summary.avg_similarity || 0) * 100).toFixed(0)}%
          </span>
          <span className="metric-label">평균 유사도</span>
        </div>
        
        <div className="metric">
          <span className="metric-value" style={{ color: getSimilarityColor(summary.max_similarity || 0) }}>
            {((summary.max_similarity || 0) * 100).toFixed(0)}%
          </span>
          <span className="metric-label">최고 유사도</span>
        </div>
        
        <div className="metric">
          <span className="metric-value" style={{ color: '#22c55e' }}>
            {highCount}/{resultCount}
          </span>
          <span className="metric-label">신뢰도 높음</span>
        </div>
        
        <div className="metric confidence-bar">
          <div className="bar-track">
            <div className="bar-fill high" style={{ width: `${highPercent}%` }}></div>
          </div>
          <span className="metric-label">신뢰도 분포</span>
        </div>
      </div>
      
      {warning && (
        <div className="quality-warning-box">
          {warning}
        </div>
      )}
      
      {summary.threshold_used && (
        <div className="threshold-info">
          임계값: {(summary.threshold_used * 100).toFixed(0)}% 이상
        </div>
      )}
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════
// 검색 결과 카드 컴포넌트 (개선됨) ← UPDATED
// ═══════════════════════════════════════════════════════════════════════════

interface ResultCardProps {
  result: SearchResult
  index: number
  onAskChunk: (idx: number, text: string) => void
}

const ResultCard = ({ result, index, onAskChunk }: ResultCardProps) => {
  const [showMetadata, setShowMetadata] = useState(false)
  const confidenceInfo = getConfidenceInfo(result.confidence)
  
  // 조항 정보 생성
  const getArticleDisplay = () => {
    const { metadata } = result
    if (!metadata.article_num) return null
    
    const typeLabel = getArticleTypeLabel(metadata.article_type)
    return `제${metadata.article_num}${typeLabel}`
  }

  return (
    <div className={`result-card confidence-${result.confidence || 'unknown'}`}>
      {/* 헤더 */}
      <div className="card-header">
        <div className="source-info">
          <span className="source-file">📄 {result.metadata?.doc_name}</span>
          {getArticleDisplay() && (
            <span className="article-info">📌 {getArticleDisplay()}</span>
          )}
          {result.metadata?.page && (
            <span className="page-info">📃 {result.metadata.page}p</span>
          )}
        </div>
        
        <div className="relevance-info">
          {/* 신뢰도 배지 */}
          <div className="confidence-badge" style={{ borderColor: confidenceInfo.color }}>
            <span className="confidence-emoji">{confidenceInfo.emoji}</span>
            <span className="confidence-label">{confidenceInfo.label}</span>
          </div>
          
          {/* 유사도 */}
          <div className="similarity-score" style={{ color: getSimilarityColor(result.similarity) }}>
            <span className="score-value">{(result.similarity * 100).toFixed(0)}%</span>
            <span className="score-label">{getSimilarityLabel(result.similarity)}</span>
          </div>
        </div>
      </div>
      
      {/* 청크 인덱스 표시 */}
      <div className="chunk-position">
        청크 {(result.metadata?.chunk_index || 0) + 1}
        {result.metadata?.total_chunks && ` / ${result.metadata.total_chunks}`}
        {result.metadata?.chunk_method && (
          <span className="chunk-method-badge">{result.metadata.chunk_method}</span>
        )}
      </div>
      
      {/* 본문 */}
      <div className="card-content">{result.text}</div>
      
      {/* 메타데이터 뷰어 */}
      <MetadataViewer 
        metadata={result.metadata} 
        isExpanded={showMetadata}
        onToggle={() => setShowMetadata(!showMetadata)}
      />
      
      {/* AI 답변 버튼 */}
      <button 
        className="chunk-ai-btn" 
        onClick={() => onAskChunk(index, result.text)} 
        disabled={result.aiLoading}
      >
        {result.aiLoading ? '생성 중...' : '🤖 이 내용으로 답변 생성'}
      </button>
      
      {/* AI 답변 */}
      {result.aiAnswer && (
        <div className="chunk-answer">
          <div className="chunk-answer-title">💡 AI 답변</div>
          <div className="chunk-answer-text">{result.aiAnswer}</div>
        </div>
      )}
    </div>
  )
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
  const [ollamaStatus, setOllamaStatus] = useState({ running: false, models: [] as string[] })
  
  // 청킹 설정
  const [chunkMethod, setChunkMethod] = useState('article')
  const [chunkSize, setChunkSize] = useState(300) // 기본값 300으로 변경 ← UPDATED
  const [semanticThreshold, setSemanticThreshold] = useState(0.5)
  const [chunkLlmModel, setChunkLlmModel] = useState('qwen2.5:3b')
  
  // 검색 설정 ← NEW
  const [similarityThreshold, setSimilarityThreshold] = useState<number | null>(null)
  const [showLowConfidence, setShowLowConfidence] = useState(true)
  
  // 에이전트
  const [enableClarification, setEnableClarification] = useState(true)
  const [clarificationMessage, setClarificationMessage] = useState<string | null>(null)
  const [clarificationOptions, setClarificationOptions] = useState<ClarificationOption[]>([])
  const [selectedDocFilter, setSelectedDocFilter] = useState<string | null>(null)
  
  // AI 답변
  const [globalAnswer, setGlobalAnswer] = useState<string>('')
  const [globalAnswerLoading, setGlobalAnswerLoading] = useState(false)
  
  // 모델 정보 팝업
  const [showModelInfo, setShowModelInfo] = useState(false)
  const [embeddingModels, setEmbeddingModels] = useState<EmbeddingModelsResponse | null>(null)

  // UI
  const [activeTab, setActiveTab] = useState<'compare' | 'multi' | 'matrix' | 'rag'>('rag')
  const [loading, setLoading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Ollama 상태 확인
  useEffect(() => {
    checkOllamaStatus()
    fetchDocuments()
    fetchEmbeddingModels()
  }, [])

  const checkOllamaStatus = async () => {
    try {
      const res = await fetch(`${API_URL}/models/llm`)
      const data: LLMModelsResponse = await res.json()
      setOllamaStatus({
        running: data.ollama?.server_running || false,
        models: data.ollama?.available_models || []
      })
      if (!data.ollama?.server_running) {
        setLlmBackend('huggingface')
        setLlmModel('Qwen/Qwen2.5-0.5B-Instruct')
      }
    } catch (e) {
      setOllamaStatus({ running: false, models: [] })
      setLlmBackend('huggingface')
    }
  }

  const fetchEmbeddingModels = async () => {
    try {
      const res = await fetch(`${API_URL}/models/embedding`)
      const data: EmbeddingModelsResponse = await res.json()
      setEmbeddingModels(data)
    } catch (e) {
      console.error('Failed to fetch embedding models:', e)
    }
  }

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_URL}/rag/documents?collection=documents`)
      const data = await res.json()
      setDocuments(data.documents || [])
    } catch (e) {
      console.error('Failed to fetch documents:', e)
    }
  }

  // 파일 업로드
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setLoading(true)
    setUploadStatus('업로드 중...')

    const formData = new FormData()
    formData.append('file', file)
    formData.append('collection', 'documents')
    formData.append('model', ragModel)
    formData.append('chunk_method', chunkMethod)
    formData.append('chunk_size', chunkSize.toString())

    try {
      const res = await fetch(`${API_URL}/rag/upload`, { method: 'POST', body: formData })
      const data = await res.json()
      
      if (data.success) {
        setUploadStatus(`✅ ${data.filename} 업로드 완료 (${data.chunks_created}개 청크, ${data.chunk_method} 방식)`)
        fetchDocuments()
      } else {
        setUploadStatus(`❌ 업로드 실패: ${data.detail || '알 수 없는 오류'}`)
      }
    } catch (e) {
      setUploadStatus(`❌ 업로드 오류: ${e}`)
    }

    setLoading(false)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  // 문서 삭제
  const handleDeleteDocument = async (docName: string) => {
    try {
      await fetch(`${API_URL}/rag/document`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_name: docName, collection: 'documents' })
      })
      fetchDocuments()
    } catch (e) {
      console.error('Failed to delete document:', e)
    }
  }

  // RAG 검색
  const handleRAGSearch = async () => {
    if (!ragQuery.trim()) return

    setLoading(true)
    setRagResult(null)
    setGlobalAnswer('')
    setClarificationMessage(null)
    setClarificationOptions([])

    try {
      const res = await fetch(`${API_URL}/rag/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: ragQuery,
          collection: 'documents',
          n_results: 5,
          model: ragModel,
          filter_doc: selectedDocFilter,
          similarity_threshold: similarityThreshold
        })
      })
      const data = await res.json()
      
      // 낮은 신뢰도 필터링 ← NEW
      let filteredResults = data.results || []
      if (!showLowConfidence) {
        filteredResults = filteredResults.filter((r: SearchResult) => r.confidence !== 'low')
      }
      
      setRagResult({ 
        query: ragQuery, 
        results: filteredResults,
        quality_summary: data.quality_summary
      })
    } catch (e) {
      console.error('Search failed:', e)
    }

    setLoading(false)
  }

  // AI 답변 (에이전트)
  const handleAIAnswer = async () => {
    if (!ragQuery.trim()) return

    setGlobalAnswerLoading(true)
    setGlobalAnswer('')

    try {
      const res = await fetch(`${API_URL}/rag/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: ragQuery,
          collection: 'documents',
          n_results: 5,
          embedding_model: ragModel,
          llm_model: llmModel,
          llm_backend: llmBackend,
          check_clarification: enableClarification,
          filter_doc: selectedDocFilter,
          similarity_threshold: similarityThreshold
        })
      })
      const data: RAGResponse = await res.json()

      if (data.needs_clarification && data.clarification_options) {
        setClarificationMessage(data.answer || '')
        setClarificationOptions(data.clarification_options)
        
        // 낮은 신뢰도 필터링
        let sources = data.sources || []
        if (!showLowConfidence) {
          sources = sources.filter(r => r.confidence !== 'low')
        }
        setRagResult({ 
          query: ragQuery, 
          results: sources,
          quality_summary: data.quality_summary,
          quality_warning: data.quality_warning
        })
      } else {
        setGlobalAnswer(data.answer || '')
        setClarificationMessage(null)
        setClarificationOptions([])
        
        // 낮은 신뢰도 필터링
        let sources = data.sources || []
        if (!showLowConfidence) {
          sources = sources.filter(r => r.confidence !== 'low')
        }
        setRagResult({ 
          query: ragQuery, 
          results: sources,
          quality_summary: data.quality_summary,
          quality_warning: data.quality_warning
        })
      }
    } catch (e) {
      setGlobalAnswer(`오류가 발생했습니다: ${e}`)
    }

    setGlobalAnswerLoading(false)
  }

  // 특정 문서 선택
  const handleSelectDocument = async (docName: string) => {
    setSelectedDocFilter(docName)
    setClarificationMessage(null)
    setClarificationOptions([])
    
    setGlobalAnswerLoading(true)
    
    try {
      const res = await fetch(`${API_URL}/rag/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: ragQuery,
          collection: 'documents',
          n_results: 5,
          embedding_model: ragModel,
          llm_model: llmModel,
          llm_backend: llmBackend,
          check_clarification: false,
          filter_doc: docName,
          similarity_threshold: similarityThreshold
        })
      })
      const data: RAGResponse = await res.json()
      
      setGlobalAnswer(data.answer || '')
      
      let sources = data.sources || []
      if (!showLowConfidence) {
        sources = sources.filter(r => r.confidence !== 'low')
      }
      setRagResult({ 
        query: ragQuery, 
        results: sources,
        quality_summary: data.quality_summary,
        quality_warning: data.quality_warning
      })
    } catch (e) {
      setGlobalAnswer(`오류: ${e}`)
    }
    
    setGlobalAnswerLoading(false)
    setSelectedDocFilter(null)
  }

  // 개별 청크 AI 답변
  const handleChunkAIAnswer = async (idx: number, chunkText: string) => {
    if (!ragResult?.results) return

    const updatedResults = [...ragResult.results]
    updatedResults[idx] = { ...updatedResults[idx], aiLoading: true }
    setRagResult({ ...ragResult, results: updatedResults })

    try {
      const res = await fetch(`${API_URL}/rag/ask-chunk`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: ragQuery,
          chunk_text: chunkText,
          llm_model: llmModel,
          llm_backend: llmBackend
        })
      })
      const data = await res.json()

      updatedResults[idx] = {
        ...updatedResults[idx],
        aiLoading: false,
        aiAnswer: data.answer
      }
      setRagResult({ ...ragResult, results: updatedResults })
    } catch (e) {
      updatedResults[idx] = {
        ...updatedResults[idx],
        aiLoading: false,
        aiAnswer: `오류: ${e}`
      }
      setRagResult({ ...ragResult, results: updatedResults })
    }
  }

  // 텍스트 비교
  const handleCompare = async () => {
    if (!text1.trim() || !text2.trim()) return
    setLoading(true)
    try {
      const res = await fetch(`${API_URL}/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text1, text2, model: selectedModel })
      })
      const data: CompareResult = await res.json()
      setResult(data)
    } catch (e) {
      console.error('Compare failed:', e)
    }
    setLoading(false)
  }

  // 모델 선택 렌더링
  const renderModelSelect = (value: string, onChange: (v: string) => void, showIncompatible = true) => (
    <div className="model-select-wrapper">
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {PRESET_MODELS.filter(m => showIncompatible || m.compatible).map(m => (
          <option key={m.key} value={m.key} disabled={!m.compatible}>
            {m.name} - {m.desc} {!m.compatible && '⚠️'}
          </option>
        ))}
      </select>
      <button className="info-btn" onClick={() => setShowModelInfo(!showModelInfo)}>ℹ️</button>
    </div>
  )

  return (
    <div className="app">
      <header className="header">
        <h1 className="title">🔍 텍스트 유사도 + RAG</h1>
        <p className="subtitle">v5.1 - 검색 품질 시각화 + 메타데이터 뷰어</p>
      </header>

      <div className="tabs">
        <button className={`tab ${activeTab === 'compare' ? 'active' : ''}`} onClick={() => setActiveTab('compare')}>📊 1:1 비교</button>
        <button className={`tab ${activeTab === 'multi' ? 'active' : ''}`} onClick={() => setActiveTab('multi')}>📈 멀티모델</button>
        <button className={`tab ${activeTab === 'matrix' ? 'active' : ''}`} onClick={() => setActiveTab('matrix')}>🧮 유사도 행렬</button>
        <button className={`tab ${activeTab === 'rag' ? 'active' : ''}`} onClick={() => setActiveTab('rag')}>💬 RAG 질문</button>
      </div>

      <main className="main">
        {activeTab === 'compare' && (
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
            <div className="model-select">
              <label>🤖 임베딩 모델</label>
              {renderModelSelect(selectedModel, setSelectedModel)}
            </div>
            <button className="primary-btn" onClick={handleCompare} disabled={loading || !text1 || !text2}>{loading ? '분석 중...' : '유사도 분석'}</button>
            {result && (
              <div className="result-box">
                <div className="score-big" style={{ color: getSimilarityColor(result.similarity) }}>{(result.similarity * 100).toFixed(1)}%</div>
                <div className="score-label">{result.interpretation}</div>
                <div className="score-bar"><div className="score-fill" style={{ width: `${result.similarity * 100}%`, backgroundColor: getSimilarityColor(result.similarity) }}></div></div>
              </div>
            )}
          </>
        )}

        {activeTab === 'multi' && (
          <>
            <div className="input-section">
              <div className="text-input"><label>텍스트 1</label><textarea value={text1} onChange={(e) => setText1(e.target.value)} placeholder="첫 번째 텍스트..." rows={4} /></div>
              <div className="text-input"><label>텍스트 2</label><textarea value={text2} onChange={(e) => setText2(e.target.value)} placeholder="두 번째 텍스트..." rows={4} /></div>
            </div>
            <div className="model-grid">
              {PRESET_MODELS.map(m => (
                <label key={m.key} className={`model-chip ${selectedModels.includes(m.key) ? 'selected' : ''} ${!m.compatible ? 'incompatible' : ''}`}>
                  <input type="checkbox" checked={selectedModels.includes(m.key)} onChange={(e) => setSelectedModels(e.target.checked ? [...selectedModels, m.key] : selectedModels.filter(k => k !== m.key))} disabled={!m.compatible} />
                  {m.name}
                </label>
              ))}
            </div>
            <button className="primary-btn" onClick={async () => {
              if (!text1.trim() || !text2.trim() || selectedModels.length === 0) return
              setLoading(true)
              try {
                const res = await fetch(`${API_URL}/compare/multi`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ text1, text2, models: selectedModels.filter(k => PRESET_MODELS.find(m => m.key === k)?.compatible) })
                })
                const data: MultiModelResult = await res.json()
                setMultiResult(data)
              } catch (e) { console.error(e) }
              setLoading(false)
            }} disabled={loading || !text1 || !text2 || selectedModels.length === 0}>{loading ? '분석 중...' : '멀티모델 비교'}</button>
            {multiResult && (
              <div className="results-list">
                {multiResult.results.map((r, i) => (
                  <div key={i} className="result-row">
                    <span className="result-name">{PRESET_MODELS.find(m => m.key === r.model_key)?.name || r.model_key}</span>
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
              {texts.map((t, i) => (
                <div key={i} className="matrix-row">
                  <span className="row-num">{i + 1}</span>
                  <textarea value={t} onChange={(e) => { const newTexts = [...texts]; newTexts[i] = e.target.value; setTexts(newTexts) }} placeholder={`텍스트 ${i + 1}`} rows={2} />
                  {texts.length > 2 && <button className="remove-btn" onClick={() => setTexts(texts.filter((_, j) => j !== i))}>×</button>}
                </div>
              ))}
              {texts.length < 6 && <button className="add-btn" onClick={() => setTexts([...texts, ''])}>+ 텍스트 추가</button>}
            </div>
            <div className="model-select"><label>🤖 임베딩 모델</label>{renderModelSelect(selectedModel, setSelectedModel)}</div>
            <button className="primary-btn" onClick={async () => {
              const validTexts = texts.filter(t => t.trim())
              if (validTexts.length < 2) return
              setLoading(true)
              try {
                const res = await fetch(`${API_URL}/matrix`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ texts: validTexts, model: selectedModel })
                })
                const data = await res.json()
                setMatrixResult({ similarity_matrix: data.matrix, texts: validTexts, model_used: data.model_used })
              } catch (e) { console.error(e) }
              setLoading(false)
            }} disabled={loading || texts.filter(t => t.trim()).length < 2}>{loading ? '분석 중...' : '유사도 행렬 생성'}</button>
            {matrixResult && (
              <div className="matrix-table-wrap">
                <table className="matrix-table">
                  <thead>
                    <tr><th></th>{matrixResult.texts.map((_, i) => <th key={i}>{i + 1}</th>)}</tr>
                  </thead>
                  <tbody>
                    {matrixResult.similarity_matrix.map((row, i) => (
                      <tr key={i}>
                        <td className="row-head">{i + 1}</td>
                        {row.map((v, j) => <td key={j} style={{ backgroundColor: `rgba(37, 99, 235, ${v * 0.5})` }}>{(v * 100).toFixed(0)}%</td>)}
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
            {showModelInfo && embeddingModels && (
              <div className="model-info-popup">
                <div className="popup-header">
                  <h4>📊 임베딩 모델 호환성 (dim≤{embeddingModels.filter_criteria.max_dim}, mem≤{embeddingModels.filter_criteria.max_memory_mb}MB)</h4>
                  <button onClick={() => setShowModelInfo(false)}>×</button>
                </div>
                <div className="model-lists">
                  <div className="compatible-list">
                    <h5>✅ 호환 ({embeddingModels.compatible.length})</h5>
                    {embeddingModels.compatible.map(m => (
                      <div key={m.path} className="model-item">
                        <span>{m.name}</span>
                        <span className="model-spec">dim:{m.dim} / {m.memory_mb}MB</span>
                      </div>
                    ))}
                  </div>
                  <div className="incompatible-list">
                    <h5>⚠️ 비호환 ({embeddingModels.incompatible.length})</h5>
                    {embeddingModels.incompatible.map(m => (
                      <div key={m.path} className="model-item warning">
                        <span>{m.name}</span>
                        <span className="model-spec">dim:{m.dim} / {m.memory_mb}MB</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

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

            {/* 청킹 설정 */}
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
                <input type="range" min="100" max="1000" step="50" value={chunkSize} onChange={(e) => setChunkSize(Number(e.target.value))} />
              </div>
              
              {chunkMethod === 'semantic' && (
                <div className="semantic-options">
                  <span>🧠 유사도 임계값: {semanticThreshold.toFixed(2)}</span>
                  <input type="range" min="0.3" max="0.8" step="0.05" value={semanticThreshold} onChange={(e) => setSemanticThreshold(Number(e.target.value))} />
                  <span className="hint">낮을수록 더 작게 분할</span>
                </div>
              )}
              
              {chunkMethod === 'llm' && (
                <div className="llm-chunk-options">
                  <span>🤖 파싱용 LLM:</span>
                  <select value={chunkLlmModel} onChange={(e) => setChunkLlmModel(e.target.value)}>
                    {llmBackend === 'ollama' ? OLLAMA_MODELS.map(m => <option key={m.key} value={m.key}>{m.name}</option>) : HF_MODELS.map(m => <option key={m.key} value={m.key}>{m.name}</option>)}
                  </select>
                  <span className="hint">⚠️ 가장 느리지만 가장 정확</span>
                </div>
              )}
            </div>

            {/* 검색 품질 설정 ← NEW */}
            <div className="search-quality-settings">
              <label className="settings-label">🎯 검색 품질 설정</label>
              <div className="quality-options">
                <div className="quality-option">
                  <label>
                    <input 
                      type="checkbox" 
                      checked={showLowConfidence} 
                      onChange={(e) => setShowLowConfidence(e.target.checked)} 
                    />
                    🔴 낮은 신뢰도 결과 표시
                  </label>
                </div>
                <div className="quality-option threshold">
                  <span>유사도 임계값: {similarityThreshold ? `${(similarityThreshold * 100).toFixed(0)}%` : '없음'}</span>
                  <input 
                    type="range" 
                    min="0" 
                    max="0.7" 
                    step="0.05" 
                    value={similarityThreshold || 0} 
                    onChange={(e) => {
                      const val = Number(e.target.value)
                      setSimilarityThreshold(val > 0 ? val : null)
                    }} 
                  />
                </div>
              </div>
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
                <label>📚 업로드된 문서 ({documents.length})</label>
                {documents.map((doc, i) => (
                  <div key={i} className="doc-item">
                    <div className="doc-info">
                      <strong>{doc.doc_name}</strong>
                      <span className="doc-meta">
                        {doc.chunk_count}개 청크
                        {doc.chunk_method && ` • ${doc.chunk_method}`}
                        {doc.model && ` • ${doc.model.split('/').pop()}`}
                      </span>
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

            {/* 되묻기 박스 */}
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

            {/* AI 종합 답변 */}
            {(globalAnswerLoading || globalAnswer) && !clarificationMessage && (
              <div className="global-answer">
                <h3>🤖 AI 종합 답변</h3>
                {globalAnswerLoading ? (
                  <div className="loading-answer">
                    <span className="spinner"></span>
                    답변 생성 중... ({llmBackend === 'ollama' ? 'Ollama' : 'HuggingFace'})
                  </div>
                ) : (
                  <div className="answer-text">{globalAnswer}</div>
                )}
              </div>
            )}

            {/* 품질 요약 카드 ← NEW */}
            {ragResult?.quality_summary && ragResult.results && ragResult.results.length > 0 && (
              <QualitySummaryCard 
                summary={ragResult.quality_summary} 
                warning={ragResult.quality_warning}
                resultCount={ragResult.results.length}
              />
            )}

            {/* 검색 결과 */}
            {ragResult?.results && ragResult.results.length > 0 && (
              <div className="search-results">
                <h3>📄 관련 문서 조각 ({ragResult.results.length}개)</h3>
                {ragResult.results.map((r, idx) => (
                  <ResultCard 
                    key={idx}
                    result={r}
                    index={idx}
                    onAskChunk={handleChunkAIAnswer}
                  />
                ))}
              </div>
            )}

            {ragResult && (!ragResult.results || ragResult.results.length === 0) && !loading && (
              <div className="no-results">관련 문서를 찾을 수 없습니다.</div>
            )}
          </>
        )}
      </main>

      <footer className="footer">v5.1 - 검색 품질 시각화 + 메타데이터 뷰어 + 신뢰도 표시</footer>
    </div>
  )
}

export default App