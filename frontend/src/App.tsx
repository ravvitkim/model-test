import { useState } from 'react'
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

// ═══════════════════════════════════════════════════════════════════════════
// 프리셋 모델
// ═══════════════════════════════════════════════════════════════════════════

const PRESET_MODELS = [
  { key: 'ko-sroberta', name: 'Ko-SROBERTA', desc: '한국어 전용 (추천)', category: 'korean' },
  { key: 'ko-sbert', name: 'Ko-SBERT', desc: '한국어 STS', category: 'korean' },
  { key: 'ko-simcse', name: 'Ko-SimCSE', desc: '한국어 SimCSE', category: 'korean' },
  { key: 'qwen3-0.6b', name: 'Qwen3-Embedding-0.6B', desc: '다국어 (가벼움)', category: 'multilingual' },
  { key: 'qwen3-4b', name: 'Qwen3-Embedding-4B', desc: '다국어 (고성능)', category: 'multilingual' },
  { key: 'multilingual-minilm', name: 'Multilingual MiniLM', desc: '다국어 (가벼움)', category: 'multilingual' },
  { key: 'multilingual-e5', name: 'Multilingual E5', desc: '다국어 (고성능)', category: 'multilingual' },
  { key: 'bge-m3', name: 'BGE-M3', desc: '다국어 (최신)', category: 'multilingual' },
  { key: 'minilm', name: 'MiniLM', desc: '영어 전용 (빠름)', category: 'english' },
  { key: 'mpnet', name: 'MPNet', desc: '영어 전용 (고성능)', category: 'english' },
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

const getCategoryColor = (category: string) => {
  switch (category) {
    case 'korean': return { bg: 'rgba(59, 130, 246, 0.3)', text: '#60a5fa' }
    case 'multilingual': return { bg: 'rgba(168, 85, 247, 0.3)', text: '#a78bfa' }
    case 'english': return { bg: 'rgba(34, 197, 94, 0.3)', text: '#4ade80' }
    default: return { bg: 'rgba(100, 116, 139, 0.3)', text: '#94a3b8' }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 메인 컴포넌트
// ═══════════════════════════════════════════════════════════════════════════

function App() {
  // 단일 비교용
  const [text1, setText1] = useState('')
  const [text2, setText2] = useState('')
  const [selectedModel, setSelectedModel] = useState('ko-sroberta')
  const [result, setResult] = useState<CompareResult | null>(null)
  
  // 다중 모델 비교용
  const [multiResult, setMultiResult] = useState<MultiModelResult | null>(null)
  const [selectedModels, setSelectedModels] = useState<string[]>(['ko-sroberta', 'qwen3-0.6b'])
  const [customModel, setCustomModel] = useState('')
  const [customModels, setCustomModels] = useState<string[]>([])
  
  // 다중 텍스트 비교용 (매트릭스)
  const [texts, setTexts] = useState<string[]>(['', '', ''])
  const [matrixResult, setMatrixResult] = useState<MatrixResult | null>(null)
  const [matrixModel, setMatrixModel] = useState('ko-sroberta')
  
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'single' | 'multi' | 'matrix'>('single')

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

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'API 요청 실패')
      }
      
      const data = await response.json()
      setResult(data)
    } catch (error) {
      alert(`오류: ${error instanceof Error ? error.message : '서버 연결 실패'}`)
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

    const allModels = [...selectedModels, ...customModels]
    if (allModels.length < 1) {
      alert('최소 1개 모델을 선택해주세요.')
      return
    }

    setLoading(true)
    setMultiResult(null)

    try {
      const response = await fetch(`${API_URL}/compare/models`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text1, text2, models: allModels })
      })

      if (!response.ok) throw new Error('API 요청 실패')
      
      const data = await response.json()
      setMultiResult(data)
    } catch (error) {
      alert('서버 연결 실패. 백엔드가 실행 중인지 확인하세요.')
    } finally {
      setLoading(false)
    }
  }

  // 다중 텍스트 매트릭스 비교
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
        body: JSON.stringify({ texts: validTexts, model: matrixModel })
      })

      if (!response.ok) throw new Error('API 요청 실패')
      
      const data = await response.json()
      setMatrixResult(data)
    } catch (error) {
      alert('서버 연결 실패. 백엔드가 실행 중인지 확인하세요.')
    } finally {
      setLoading(false)
    }
  }

  const toggleModelSelection = (modelKey: string) => {
    setSelectedModels(prev => 
      prev.includes(modelKey) 
        ? prev.filter(m => m !== modelKey)
        : [...prev, modelKey]
    )
  }

  const addCustomModel = () => {
    if (!customModel.trim()) return
    if (customModels.includes(customModel) || PRESET_MODELS.some(m => m.key === customModel)) {
      alert('이미 추가된 모델입니다.')
      return
    }
    setCustomModels(prev => [...prev, customModel])
    setCustomModel('')
  }

  const removeCustomModel = (model: string) => {
    setCustomModels(prev => prev.filter(m => m !== model))
  }

  // 텍스트 입력 필드 관리
  const updateText = (index: number, value: string) => {
    const newTexts = [...texts]
    newTexts[index] = value
    setTexts(newTexts)
  }

  const addTextField = () => {
    if (texts.length < 10) {
      setTexts([...texts, ''])
    }
  }

  const removeTextField = (index: number) => {
    if (texts.length > 2) {
      setTexts(texts.filter((_, i) => i !== index))
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1 className="title">🔍 텍스트 유사도 비교</h1>
        <p className="pipeline">
          [원문] → [파싱: 품사분석] → [청킹: 의미단위] → [임베딩: 벡터] → [코사인 유사도]
        </p>
      </header>

      <div className="tabs">
        <button 
          className={`tab ${activeTab === 'single' ? 'active' : ''}`}
          onClick={() => setActiveTab('single')}
        >
          단일 모델 비교
        </button>
        <button 
          className={`tab ${activeTab === 'multi' ? 'active' : ''}`}
          onClick={() => setActiveTab('multi')}
        >
          🔥 모델 비교
        </button>
        <button 
          className={`tab ${activeTab === 'matrix' ? 'active' : ''}`}
          onClick={() => setActiveTab('matrix')}
        >
          📊 다중 텍스트
        </button>
      </div>

      <main className="main">
        {/* ═══════════════════════════════════════════════════════════════ */}
        {/* 단일 모델 탭 */}
        {/* ═══════════════════════════════════════════════════════════════ */}
        {activeTab === 'single' && (
          <>
            <div className="input-section">
              <div className="text-input">
                <label>텍스트 1</label>
                <textarea
                  value={text1}
                  onChange={(e) => setText1(e.target.value)}
                  placeholder="첫 번째 텍스트를 입력하세요..."
                  rows={5}
                />
              </div>
              <div className="text-input">
                <label>텍스트 2</label>
                <textarea
                  value={text2}
                  onChange={(e) => setText2(e.target.value)}
                  placeholder="두 번째 텍스트를 입력하세요..."
                  rows={5}
                />
              </div>
            </div>

            <div className="model-select">
              <label>임베딩 모델 선택</label>
              <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                <optgroup label="🇰🇷 한국어 전용">
                  {PRESET_MODELS.filter(m => m.category === 'korean').map(m => (
                    <option key={m.key} value={m.key}>{m.name} - {m.desc}</option>
                  ))}
                </optgroup>
                <optgroup label="🌍 다국어">
                  {PRESET_MODELS.filter(m => m.category === 'multilingual').map(m => (
                    <option key={m.key} value={m.key}>{m.name} - {m.desc}</option>
                  ))}
                </optgroup>
                <optgroup label="🇺🇸 영어 전용">
                  {PRESET_MODELS.filter(m => m.category === 'english').map(m => (
                    <option key={m.key} value={m.key}>{m.name} - {m.desc}</option>
                  ))}
                </optgroup>
              </select>
            </div>

            <div className="custom-model-section">
              <label>또는 HuggingFace 모델 경로 직접 입력</label>
              <input
                type="text"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                placeholder="예: Qwen/Qwen3-Embedding-0.6B"
              />
            </div>

            <button className="compare-btn" onClick={handleCompare} disabled={loading}>
              {loading ? '분석 중...' : '유사도 비교'}
            </button>

            {result && (
              <div className="result-section">
                <div className="score-display">
                  <h2>유사도 점수</h2>
                  <div className="score" style={{ color: getSimilarityColor(result.similarity) }}>
                    {(result.similarity * 100).toFixed(1)}%
                  </div>
                  <div className="interpretation">{result.interpretation}</div>
                  <div className="score-bar">
                    <div 
                      className="score-fill"
                      style={{
                        width: `${Math.max(result.similarity * 100, 5)}%`,
                        backgroundColor: getSimilarityColor(result.similarity)
                      }}
                    />
                  </div>
                  <p className="model-info">
                    모델: {result.model_used}<br/>
                    로드: {result.load_time}s | 추론: {result.inference_time}s
                  </p>
                </div>

                <div className="details-grid">
                  <div className="detail-card">
                    <h3>텍스트 1 처리 결과</h3>
                    <div className="detail-item">
                      <strong>청킹 결과:</strong>
                      {result.text1_processed.chunks.map((chunk, i) => (
                        <div key={i} className="chunk">{chunk}</div>
                      ))}
                    </div>
                    <div className="detail-item">
                      <strong>품사 태그:</strong>
                      <div className="pos-tags">
                        {result.text1_processed.pos_tags.slice(0, 8).map((tag, i) => (
                          <span key={i} className="pos-tag">
                            {tag[0]}<sub>{tag[1]}</sub>
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="detail-card">
                    <h3>텍스트 2 처리 결과</h3>
                    <div className="detail-item">
                      <strong>청킹 결과:</strong>
                      {result.text2_processed.chunks.map((chunk, i) => (
                        <div key={i} className="chunk">{chunk}</div>
                      ))}
                    </div>
                    <div className="detail-item">
                      <strong>품사 태그:</strong>
                      <div className="pos-tags">
                        {result.text2_processed.pos_tags.slice(0, 8).map((tag, i) => (
                          <span key={i} className="pos-tag">
                            {tag[0]}<sub>{tag[1]}</sub>
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {/* ═══════════════════════════════════════════════════════════════ */}
        {/* 다중 모델 비교 탭 */}
        {/* ═══════════════════════════════════════════════════════════════ */}
        {activeTab === 'multi' && (
          <>
            <div className="input-section">
              <div className="text-input">
                <label>텍스트 1</label>
                <textarea
                  value={text1}
                  onChange={(e) => setText1(e.target.value)}
                  placeholder="첫 번째 텍스트를 입력하세요..."
                  rows={5}
                />
              </div>
              <div className="text-input">
                <label>텍스트 2</label>
                <textarea
                  value={text2}
                  onChange={(e) => setText2(e.target.value)}
                  placeholder="두 번째 텍스트를 입력하세요..."
                  rows={5}
                />
              </div>
            </div>

            <div className="model-multi-select">
              <label>비교할 모델 선택</label>
              <div className="model-checkboxes">
                {PRESET_MODELS.map(m => {
                  const isSelected = selectedModels.includes(m.key)
                  const catColor = getCategoryColor(m.category)
                  return (
                    <label 
                      key={m.key} 
                      className={`checkbox-label ${isSelected ? 'selected' : ''}`}
                    >
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => toggleModelSelection(m.key)}
                      />
                      <span className="checkbox-text">
                        <strong>{m.name}</strong>
                        <span 
                          className="category-badge"
                          style={{ background: catColor.bg, color: catColor.text }}
                        >
                          {m.category === 'korean' ? '한국어' : m.category === 'multilingual' ? '다국어' : '영어'}
                        </span>
                        <small>{m.desc}</small>
                      </span>
                    </label>
                  )
                })}
              </div>
            </div>

            <div className="custom-model-section">
              <label>✨ 커스텀 HuggingFace 모델 추가</label>
              <div className="custom-model-input">
                <input
                  type="text"
                  value={customModel}
                  onChange={(e) => setCustomModel(e.target.value)}
                  placeholder="예: intfloat/multilingual-e5-small"
                  onKeyDown={(e) => e.key === 'Enter' && addCustomModel()}
                />
                <button className="add-btn" onClick={addCustomModel}>추가</button>
              </div>
              
              {customModels.length > 0 && (
                <div className="custom-models-list">
                  <strong>추가된 모델:</strong>
                  <div className="custom-model-tags">
                    {customModels.map(model => (
                      <span key={model} className="custom-model-tag">
                        {model}
                        <button onClick={() => removeCustomModel(model)}>×</button>
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <button className="compare-btn" onClick={handleMultiCompare} disabled={loading}>
              {loading ? '모델 비교 중... (첫 로드시 오래 걸림)' : `${selectedModels.length + customModels.length}개 모델로 비교`}
            </button>

            {multiResult && (
              <div className="result-section">
                <h2 className="result-title">📊 모델별 유사도 비교 결과</h2>
                
                {multiResult.results.map((r, idx) => (
                  <div key={idx} className="model-result-item">
                    <div className="model-result-header">
                      <div>
                        <span className="model-rank">#{idx + 1}</span>
                        <span className="model-name">{r.model_key}</span>
                        {!r.success && <span className="error-badge">(로드 실패)</span>}
                      </div>
                      <span 
                        className="model-score"
                        style={{ color: r.success ? getSimilarityColor(r.similarity) : '#ef4444' }}
                      >
                        {r.success ? `${(r.similarity * 100).toFixed(1)}%` : 'ERROR'}
                      </span>
                    </div>
                    
                    <div className="model-path">{r.model_path}</div>
                    
                    {r.success ? (
                      <>
                        <div className="score-bar">
                          <div 
                            className="score-fill"
                            style={{
                              width: `${Math.max(r.similarity * 100, 5)}%`,
                              backgroundColor: getSimilarityColor(r.similarity)
                            }}
                          />
                        </div>
                        <div className="time-info">
                          {r.interpretation} | 로드: {r.load_time}s | 추론: {r.inference_time}s
                        </div>
                      </>
                    ) : (
                      <div className="error-message">{r.error}</div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {/* ═══════════════════════════════════════════════════════════════ */}
        {/* 다중 텍스트 매트릭스 탭 */}
        {/* ═══════════════════════════════════════════════════════════════ */}
        {activeTab === 'matrix' && (
          <>
            <div className="matrix-input-section">
              <div className="matrix-header">
                <label>비교할 텍스트들 (최소 2개, 최대 10개)</label>
                <button className="add-text-btn" onClick={addTextField} disabled={texts.length >= 10}>
                  + 텍스트 추가
                </button>
              </div>
              
              {texts.map((text, index) => (
                <div key={index} className="matrix-text-input">
                  <div className="text-number">{index + 1}</div>
                  <textarea
                    value={text}
                    onChange={(e) => updateText(index, e.target.value)}
                    placeholder={`텍스트 ${index + 1}을 입력하세요...`}
                    rows={2}
                  />
                  {texts.length > 2 && (
                    <button className="remove-text-btn" onClick={() => removeTextField(index)}>
                      ×
                    </button>
                  )}
                </div>
              ))}
            </div>

            <div className="model-select">
              <label>임베딩 모델 선택</label>
              <select value={matrixModel} onChange={(e) => setMatrixModel(e.target.value)}>
                <optgroup label="🇰🇷 한국어 전용">
                  {PRESET_MODELS.filter(m => m.category === 'korean').map(m => (
                    <option key={m.key} value={m.key}>{m.name} - {m.desc}</option>
                  ))}
                </optgroup>
                <optgroup label="🌍 다국어">
                  {PRESET_MODELS.filter(m => m.category === 'multilingual').map(m => (
                    <option key={m.key} value={m.key}>{m.name} - {m.desc}</option>
                  ))}
                </optgroup>
                <optgroup label="🇺🇸 영어 전용">
                  {PRESET_MODELS.filter(m => m.category === 'english').map(m => (
                    <option key={m.key} value={m.key}>{m.name} - {m.desc}</option>
                  ))}
                </optgroup>
              </select>
            </div>

            <button className="compare-btn" onClick={handleMatrixCompare} disabled={loading}>
              {loading ? '매트릭스 계산 중...' : `${texts.filter(t => t.trim()).length}개 텍스트 유사도 매트릭스`}
            </button>

            {matrixResult && (
              <div className="result-section">
                <h2 className="result-title">📊 유사도 매트릭스</h2>
                <p className="model-info center">모델: {matrixResult.model_used}</p>
                
                <div className="matrix-container">
                  <table className="matrix-table">
                    <thead>
                      <tr>
                        <th></th>
                        {matrixResult.texts.map((_, i) => (
                          <th key={i}>T{i + 1}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {matrixResult.similarity_matrix.map((row, i) => (
                        <tr key={i}>
                          <td className="row-header">T{i + 1}</td>
                          {row.map((score, j) => (
                            <td 
                              key={j} 
                              className="matrix-cell"
                              style={{ 
                                backgroundColor: i === j ? 'rgba(100,100,100,0.3)' : `${getSimilarityColor(score)}33`,
                                color: i === j ? '#888' : getSimilarityColor(score)
                              }}
                            >
                              {(score * 100).toFixed(1)}%
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="matrix-legend">
                  <h3>텍스트 목록</h3>
                  {matrixResult.texts.map((text, i) => (
                    <div key={i} className="legend-item">
                      <span className="legend-number">T{i + 1}</span>
                      <span className="legend-text">{text.length > 50 ? text.slice(0, 50) + '...' : text}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </main>

      <footer className="footer">
        <p>HuggingFace 임베딩 모델 기반 텍스트 유사도 비교 도구</p>
      </footer>
    </div>
  )
}

export default App