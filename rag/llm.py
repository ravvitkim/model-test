"""
LLM 모듈 - HuggingFace + Ollama 지원
"""

import torch
import requests
from typing import Dict, List, Optional, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

_loaded_llm: Dict[str, Any] = {}


# ═══════════════════════════════════════════════════════════════════════════
# Ollama 백엔드 (로컬 추천) - 네이티브 API 사용
# ═══════════════════════════════════════════════════════════════════════════

class OllamaLLM:
    """Ollama 네이티브 API 사용 (chat/generate 자동 선택)"""

    def __init__(self, model: str = "qwen2.5:3b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, system: str = None, temperature: float = 0.1, max_tokens: int = 256) -> str:
        """Ollama API 호출 - /api/chat 시도 후 실패시 /api/generate 사용"""
        
        print(f"🤖 Ollama 호출: model={self.model}, prompt 길이={len(prompt)}")
        
        # 1차 시도: /api/chat (최신 Ollama)
        try:
            return self._call_chat_api(prompt, system, temperature, max_tokens)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"⚠️ /api/chat 404 - /api/generate로 fallback")
                # /api/chat이 없으면 /api/generate 시도
                return self._call_generate_api(prompt, system, temperature, max_tokens)
            raise
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Ollama 서버에 연결할 수 없습니다. "
                f"'ollama serve' 명령으로 서버를 시작하고 "
                f"'ollama pull {self.model}' 로 모델을 다운로드하세요."
            )
    
    def _call_chat_api(self, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        """/api/chat 엔드포인트 (Ollama 0.1.14+)"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        
        # Qwen3 모델은 thinking 모드가 기본 → /no_think로 비활성화
        final_prompt = prompt
        if "qwen3" in self.model.lower():
            final_prompt = f"/no_think {prompt}"
        
        messages.append({"role": "user", "content": final_prompt})
        
        try:
            # Qwen3용 추가 옵션
            options = {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": options,
                    "think": False,  # Qwen3 thinking 모드 끄기
                },
                timeout=120
            )
            print(f"📡 Ollama /api/chat 응답 코드: {response.status_code}")
            
            if not response.ok:
                print(f"❌ Ollama 에러 응답: {response.text}")
            
            response.raise_for_status()
            data = response.json()
            print(f"✅ Ollama 응답 키: {data.keys()}")
            
            message = data.get("message", {})
            content = message.get("content", "") if isinstance(message, dict) else str(message)
            
            # Qwen3 thinking 모드: content가 비어있으면 thinking 사용
            if not content and isinstance(message, dict):
                thinking = message.get("thinking", "")
                if thinking:
                    print(f"🧠 thinking 모드 감지 - thinking 내용 사용")
                    # thinking에서 실제 답변 부분 추출 (마지막 부분이 보통 답변)
                    content = thinking
            
            print(f"📝 최종 content 길이: {len(content)}")
            
            if not content:
                print(f"⚠️ 빈 응답! 전체 data: {data}")
            
            return content
        except requests.exceptions.HTTPError:
            raise
        except Exception as e:
            print(f"❌ Ollama chat API 호출 실패: {type(e).__name__}: {e}")
            raise
    
    def _call_generate_api(self, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        """/api/generate 엔드포인트 (구버전 호환)"""
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                },
                timeout=120
            )
            print(f"📡 Ollama /api/generate 응답 코드: {response.status_code}")
            
            if not response.ok:
                print(f"❌ Ollama 에러 응답: {response.text}")
            
            response.raise_for_status()
            data = response.json()
            print(f"✅ Ollama 응답 키: {data.keys()}")
            return data.get("response", "")
        except Exception as e:
            print(f"❌ Ollama API 호출 실패: {type(e).__name__}: {e}")
            raise

    @staticmethod
    def list_models(base_url: str = "http://localhost:11434") -> List[str]:
        """사용 가능한 모델 목록"""
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.ok:
                data = response.json()
                models = data.get("models", [])
                return [m["name"] for m in models]
        except Exception as e:
            print(f"⚠️ Ollama 모델 목록 조회 실패: {e}")
        return []

    @staticmethod
    def is_available(base_url: str = "http://localhost:11434") -> bool:
        """Ollama 서버 실행 여부"""
        try:
            # /api/tags로 확인
            response = requests.get(f"{base_url}/api/tags", timeout=3)
            if response.ok:
                return True
            # fallback: 루트 경로 확인
            response = requests.get(base_url, timeout=3)
            return response.ok
        except:
            return False


# ═══════════════════════════════════════════════════════════════════════════
# HuggingFace 백엔드
# ═══════════════════════════════════════════════════════════════════════════

def load_llm(model_name: str):
    """HuggingFace LLM 로드 (캐싱)"""
    if model_name in _loaded_llm:
        return _loaded_llm[model_name]

    print(f"🤖 Loading LLM: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # dtype 설정
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()

    _loaded_llm[model_name] = (tokenizer, model)
    print(f"✅ LLM loaded: {model_name}")
    return tokenizer, model


def generate_with_hf(
    prompt: str,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens: int = 256,
    temperature: float = 0.1
) -> str:
    """HuggingFace 모델로 텍스트 생성"""
    tokenizer, model = load_llm(model_name)
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    decoded = tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True
    )
    
    return decoded.strip()


# ═══════════════════════════════════════════════════════════════════════════
# 통합 LLM 인터페이스
# ═══════════════════════════════════════════════════════════════════════════

def get_llm_response(
    prompt: str,
    llm_model: str = "qwen2.5:3b",
    llm_backend: str = "ollama",
    max_tokens: int = 256,
    temperature: float = 0.1
) -> str:
    """
    통합 LLM 응답 생성
    
    Args:
        prompt: 프롬프트
        llm_model: 모델명
        llm_backend: 'ollama' 또는 'huggingface'
        max_tokens: 최대 토큰
        temperature: 온도
    """
    if llm_backend == "ollama":
        llm = OllamaLLM(llm_model)
        return llm.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    else:
        return generate_with_hf(prompt, llm_model, max_tokens, temperature)


# ═══════════════════════════════════════════════════════════════════════════
# 에이전트 핵심: 되묻기 분석 및 섹션 추출
# ═══════════════════════════════════════════════════════════════════════════

def analyze_search_results(results: List[Dict]) -> Dict:
    """
    검색 결과의 메타데이터(조항, 섹션)를 분석하여 되묻기 여부 판단
    """
    if not results:
        return {'needs_clarification': False, 'options': [], 'unique_documents': []}

    doc_groups = {}

    for r in results:
        meta = r.get('metadata', {})
        doc_name = meta.get('doc_name', 'unknown')
        doc_title = meta.get('doc_title', doc_name)
        article_num = meta.get('article_num')
        article_type = meta.get('article_type', 'article')
        score = r.get('similarity', 0)
        
        # 표시용 섹션 이름 생성
        section_display = ""
        if article_num:
            if article_type == 'article': 
                section_display = f"제{article_num}조"
            elif article_type == 'chapter': 
                section_display = f"제{article_num}장"
            else: 
                section_display = f"{article_num}"

        if doc_name not in doc_groups:
            doc_groups[doc_name] = {
                'title': doc_title,
                'max_score': score,
                'sections': {section_display} if section_display else set(),
                'count': 1
            }
        else:
            doc_groups[doc_name]['max_score'] = max(doc_groups[doc_name]['max_score'], score)
            if section_display:
                doc_groups[doc_name]['sections'].add(section_display)
            doc_groups[doc_name]['count'] += 1

    unique_docs = list(doc_groups.keys())
    
    # 되묻기 판별 로직
    needs_clarification = False
    if len(unique_docs) > 1:
        scores = sorted([info['max_score'] for info in doc_groups.values()], reverse=True)
        if len(scores) >= 2 and (scores[0] - scores[1]) < 0.15:
            needs_clarification = True

    # 선택지 데이터 구성
    clarification_options = []
    for d_name in unique_docs:
        info = doc_groups[d_name]
        sections_list = sorted(list(info['sections']))
        sections_str = f" ({', '.join(sections_list[:2])})" if sections_list else ""
        
        clarification_options.append({
            "doc_name": d_name,
            "doc_title": info['title'],
            "display_text": f"{info['title']}{sections_str}",
            "score": info['max_score']
        })

    return {
        'needs_clarification': needs_clarification,
        'options': clarification_options,
        'unique_documents': unique_docs
    }


def generate_clarification_question(
    query: str, 
    options: List[Dict],
    llm_model: str = "qwen2.5:3b",
    llm_backend: str = "ollama"
) -> str:
    """
    섹션 정보를 포함하여 사용자에게 던질 질문 생성
    """
    options_text = "\n".join([f"- {opt['display_text']}" for opt in options])
    
    prompt = f"""사용자가 "{query}"에 대해 질문했습니다.
관련하여 다음 문서들의 특정 조항들이 검색되었습니다:
{options_text}

사용자에게 어떤 문서(SOP)의 내용을 바탕으로 답변을 드릴지 정중하게 물어보세요.
검색된 조항들(예: 제X조)을 언급하여 전문성을 보여주세요. 
답변은 반드시 한국어로 짧고 명확하게 하세요."""
    
    try:
        return get_llm_response(prompt, llm_model, llm_backend, max_tokens=200)
    except:
        return f"'{query}'에 대해 여러 규정(SOP)이 발견되었습니다. 어떤 문서의 내용을 확인해 드릴까요?\n\n" + options_text


# ═══════════════════════════════════════════════════════════════════════════
# 최종 답변 생성 (RAG)
# ═══════════════════════════════════════════════════════════════════════════

def generate_answer_with_context(
    query: str,
    context: str,
    llm_model: str = "qwen2.5:3b",
    llm_backend: str = "ollama"
) -> str:
    """컨텍스트를 기반으로 최종 답변 생성"""
    prompt = f"""당신은 규정(SOP) 전문가입니다. 아래 제공된 [참고 문서]의 내용을 바탕으로 사용자의 질문에 답변하세요.
문서에 없는 내용은 추측하지 마세요. 답변 시 근거가 되는 조항(예: 제N조)이 있다면 반드시 언급하세요.

[참고 문서]
{context}

[사용자 질문]
{query}

[전문가 답변]:"""
    
    return get_llm_response(prompt, llm_model, llm_backend, max_tokens=1024, temperature=0.2)


# ═══════════════════════════════════════════════════════════════════════════
# 모델 프리셋
# ═══════════════════════════════════════════════════════════════════════════

OLLAMA_MODELS = [
    {"key": "qwen2.5:0.5b", "name": "Qwen2.5-0.5B", "desc": "초경량 (1GB)", "vram": "1GB"},
    {"key": "qwen2.5:1.5b", "name": "Qwen2.5-1.5B", "desc": "경량 (2GB)", "vram": "2GB"},
    {"key": "qwen2.5:3b", "name": "Qwen2.5-3B", "desc": "추천 (3GB)", "vram": "3GB"},
    {"key": "qwen2.5:7b", "name": "Qwen2.5-7B", "desc": "고성능 (5GB)", "vram": "5GB"},
    {"key": "qwen3:4b", "name": "Qwen3-4B", "desc": "최신 추천 (4GB)", "vram": "4GB"},
    {"key": "llama3.2:3b", "name": "Llama3.2-3B", "desc": "경량 (3GB)", "vram": "3GB"},
    {"key": "gemma2:2b", "name": "Gemma2-2B", "desc": "경량 (2GB)", "vram": "2GB"},
    {"key": "gemma2:9b", "name": "Gemma2-9B", "desc": "고성능 (6GB)", "vram": "6GB"},
    {"key": "mistral:7b", "name": "Mistral-7B", "desc": "영어 특화 (5GB)", "vram": "5GB"},
]

HUGGINGFACE_MODELS = [
    {"key": "Qwen/Qwen2.5-0.5B-Instruct", "name": "Qwen2.5-0.5B", "desc": "초경량"},
    {"key": "Qwen/Qwen2.5-1.5B-Instruct", "name": "Qwen2.5-1.5B", "desc": "경량"},
    {"key": "Qwen/Qwen2.5-3B-Instruct", "name": "Qwen2.5-3B", "desc": "VRAM 6GB+"},
    {"key": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "name": "TinyLlama", "desc": "영어 특화"},
]


