"""
RAG 에이전트 v6.0
- 다단계 검색 및 추론
- 도구 사용 (검색, 요약, 비교)
- 되묻기 및 명확화
"""

from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import re


class AgentAction(Enum):
    """에이전트 액션 타입"""
    SEARCH = "search"           # 벡터 검색
    CLARIFY = "clarify"         # 되묻기
    ANSWER = "answer"           # 최종 답변
    SUMMARIZE = "summarize"     # 요약
    COMPARE = "compare"         # 비교
    REFINE = "refine"           # 검색 정제
    NONE = "none"               # 액션 없음


@dataclass
class AgentState:
    """에이전트 상태"""
    query: str
    original_query: str
    search_results: List[Dict] = field(default_factory=list)
    context: str = ""
    answer: str = ""
    needs_clarification: bool = False
    clarification_options: List[Dict] = field(default_factory=list)
    selected_doc: Optional[str] = None
    history: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    iteration: int = 0
    max_iterations: int = 5


@dataclass
class AgentResponse:
    """에이전트 응답"""
    answer: str
    sources: List[Dict]
    needs_clarification: bool = False
    clarification_options: List[Dict] = field(default_factory=list)
    action_taken: str = ""
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "needs_clarification": self.needs_clarification,
            "clarification_options": self.clarification_options,
            "action_taken": self.action_taken,
            "metadata": self.metadata,
        }


class RAGAgent:
    """
    RAG 에이전트
    
    기능:
    - 질문 분석 및 검색 전략 수립
    - 다단계 검색 및 리랭킹
    - 되묻기 판단
    - 컨텍스트 기반 답변 생성
    """

    def __init__(
        self,
        search_fn: Callable[[str, int, Optional[str]], tuple],
        llm_fn: Callable[[str], str],
        analyze_fn: Callable[[List[Dict]], Dict],
        clarify_fn: Callable[[str, List[Dict]], str],
        similarity_threshold: float = 0.35,
        enable_clarification: bool = True,
    ):
        """
        Args:
            search_fn: 검색 함수 (query, n_results, filter_doc) -> (results, context)
            llm_fn: LLM 응답 함수 (prompt) -> response
            analyze_fn: 검색 결과 분석 함수 (results) -> analysis
            clarify_fn: 되묻기 생성 함수 (query, options) -> question
            similarity_threshold: 유사도 임계값
            enable_clarification: 되묻기 활성화 여부
        """
        self.search_fn = search_fn
        self.llm_fn = llm_fn
        self.analyze_fn = analyze_fn
        self.clarify_fn = clarify_fn
        self.similarity_threshold = similarity_threshold
        self.enable_clarification = enable_clarification

    def run(
        self,
        query: str,
        n_results: int = 5,
        filter_doc: Optional[str] = None,
    ) -> AgentResponse:
        """에이전트 실행"""
        state = AgentState(
            query=query,
            original_query=query,
            selected_doc=filter_doc,
        )

        while state.iteration < state.max_iterations:
            state.iteration += 1
            action = self._decide_action(state)

            if action == AgentAction.SEARCH:
                self._execute_search(state, n_results)

            elif action == AgentAction.CLARIFY:
                return self._create_clarification_response(state)

            elif action == AgentAction.ANSWER:
                return self._create_answer_response(state)

            elif action == AgentAction.REFINE:
                self._refine_search(state, n_results)

            elif action == AgentAction.NONE:
                break

        # 최대 반복 도달 - 현재 상태로 답변
        return self._create_answer_response(state)

    def _decide_action(self, state: AgentState) -> AgentAction:
        """다음 액션 결정"""
        # 검색 결과 없으면 검색
        if not state.search_results:
            return AgentAction.SEARCH

        # 되묻기 체크
        if self.enable_clarification and not state.selected_doc:
            analysis = self.analyze_fn(state.search_results)
            if analysis['needs_clarification']:
                state.needs_clarification = True
                state.clarification_options = analysis['options']
                return AgentAction.CLARIFY

        # 결과 품질 체크
        if state.search_results:
            max_sim = max(r.get('similarity', 0) for r in state.search_results)
            if max_sim < self.similarity_threshold and state.iteration < 3:
                return AgentAction.REFINE

        return AgentAction.ANSWER

    def _execute_search(self, state: AgentState, n_results: int):
        """검색 실행"""
        results, context = self.search_fn(
            state.query,
            n_results,
            state.selected_doc
        )
        state.search_results = results
        state.context = context
        state.history.append({
            "action": "search",
            "query": state.query,
            "results_count": len(results),
        })

    def _refine_search(self, state: AgentState, n_results: int):
        """검색 정제 (쿼리 확장)"""
        # LLM으로 쿼리 확장
        expand_prompt = f"""다음 질문을 검색에 더 적합하게 확장하세요.
키워드와 동의어를 포함해주세요.

원래 질문: {state.original_query}

확장된 검색어 (한 줄로):"""

        try:
            expanded = self.llm_fn(expand_prompt).strip()
            if expanded and expanded != state.query:
                state.query = expanded
                self._execute_search(state, n_results)
        except Exception:
            pass

    def _create_clarification_response(self, state: AgentState) -> AgentResponse:
        """되묻기 응답 생성"""
        clarification_text = self.clarify_fn(
            state.original_query,
            state.clarification_options
        )

        return AgentResponse(
            answer=clarification_text,
            sources=state.search_results,
            needs_clarification=True,
            clarification_options=state.clarification_options,
            action_taken="clarify",
            metadata={"iteration": state.iteration},
        )

    def _create_answer_response(self, state: AgentState) -> AgentResponse:
        """답변 응답 생성"""
        if not state.search_results:
            return AgentResponse(
                answer="관련 문서를 찾을 수 없습니다. 질문을 다르게 표현해 보세요.",
                sources=[],
                action_taken="no_results",
            )

        # 프롬프트 구성
        prompt = self._build_answer_prompt(state)

        try:
            answer = self.llm_fn(prompt)
        except Exception as e:
            answer = f"답변 생성 중 오류: {str(e)}"

        return AgentResponse(
            answer=answer,
            sources=state.search_results,
            needs_clarification=False,
            action_taken="answer",
            metadata={
                "iteration": state.iteration,
                "history": state.history,
            },
        )

    def _build_answer_prompt(self, state: AgentState) -> str:
        """답변 프롬프트 구성"""
        return f"""당신은 규정(SOP) 전문가입니다. 아래 [참고 문서]를 바탕으로 질문에 답변하세요.

지침:
- 문서에 없는 내용은 답변하지 마세요.
- 근거가 되는 조항(예: 제N조)을 반드시 언급하세요.
- 표가 있다면 표 내용도 참조하세요.

[참고 문서]
{state.context}

[질문]
{state.original_query}

[전문가 답변]:"""


class ReActAgent:
    """
    ReAct 패턴 에이전트
    - Reasoning + Acting
    - 도구 사용 기반
    """

    SYSTEM_PROMPT = """당신은 문서 검색 및 분석 전문가입니다.
다음 도구를 사용할 수 있습니다:

1. search(query): 문서 검색
2. summarize(text): 텍스트 요약
3. compare(doc1, doc2): 두 문서 비교
4. answer(text): 최종 답변 제출

각 단계마다 다음 형식으로 응답하세요:

Thought: [현재 상황 분석 및 다음 단계 계획]
Action: [도구명](파라미터)
Observation: [도구 실행 결과 - 시스템이 채움]

최종 답변 시:
Thought: 충분한 정보를 수집했습니다.
Action: answer(최종 답변 내용)"""

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        tools: Dict[str, Callable],
        max_iterations: int = 5,
    ):
        self.llm_fn = llm_fn
        self.tools = tools
        self.max_iterations = max_iterations

    def run(self, query: str) -> AgentResponse:
        """ReAct 에이전트 실행"""
        history = []
        observations = []

        prompt = f"{self.SYSTEM_PROMPT}\n\n질문: {query}\n"

        for i in range(self.max_iterations):
            # LLM 호출
            response = self.llm_fn(prompt)
            history.append({"iteration": i, "response": response})

            # Action 파싱
            action_match = re.search(r'Action:\s*(\w+)\(([^)]*)\)', response)
            if not action_match:
                break

            action_name = action_match.group(1)
            action_param = action_match.group(2).strip('"\'')

            # 최종 답변
            if action_name == "answer":
                return AgentResponse(
                    answer=action_param,
                    sources=[],
                    action_taken="react_answer",
                    metadata={"history": history, "iterations": i + 1},
                )

            # 도구 실행
            if action_name in self.tools:
                try:
                    result = self.tools[action_name](action_param)
                    observation = f"Observation: {result}"
                except Exception as e:
                    observation = f"Observation: Error - {str(e)}"
            else:
                observation = f"Observation: Unknown tool '{action_name}'"

            observations.append(observation)
            prompt += f"\n{response}\n{observation}\n"

        return AgentResponse(
            answer="최대 반복 횟수에 도달했습니다.",
            sources=[],
            action_taken="max_iterations",
            metadata={"history": history},
        )


class PlanAndExecuteAgent:
    """
    Plan-and-Execute 패턴 에이전트
    - 먼저 계획 수립
    - 단계별 실행
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        search_fn: Callable,
        max_steps: int = 5,
    ):
        self.llm_fn = llm_fn
        self.search_fn = search_fn
        self.max_steps = max_steps

    def run(self, query: str) -> AgentResponse:
        """Plan-and-Execute 실행"""
        # 1. 계획 수립
        plan = self._create_plan(query)

        # 2. 단계별 실행
        results = []
        for i, step in enumerate(plan[:self.max_steps]):
            step_result = self._execute_step(step, query, results)
            results.append({"step": step, "result": step_result})

        # 3. 최종 답변 생성
        answer = self._synthesize_answer(query, results)

        return AgentResponse(
            answer=answer,
            sources=[],
            action_taken="plan_execute",
            metadata={"plan": plan, "results": results},
        )

    def _create_plan(self, query: str) -> List[str]:
        """계획 수립"""
        prompt = f"""다음 질문에 답하기 위한 단계별 계획을 수립하세요.
각 단계를 한 줄씩 작성하세요.

질문: {query}

계획:
1."""

        try:
            response = self.llm_fn(prompt)
            # 번호가 있는 줄 추출
            steps = re.findall(r'\d+\.\s*(.+)', response)
            return steps if steps else ["문서 검색", "정보 분석", "답변 작성"]
        except Exception:
            return ["문서 검색", "정보 분석", "답변 작성"]

    def _execute_step(
        self,
        step: str,
        query: str,
        previous_results: List[Dict]
    ) -> str:
        """단계 실행"""
        step_lower = step.lower()

        if "검색" in step_lower or "search" in step_lower:
            results, context = self.search_fn(query, 5, None)
            return context if context else "검색 결과 없음"

        elif "분석" in step_lower or "analy" in step_lower:
            prev_context = "\n".join([r.get("result", "") for r in previous_results])
            prompt = f"""다음 정보를 분석하세요:

{prev_context}

분석 결과:"""
            return self.llm_fn(prompt)

        else:
            return f"단계 실행: {step}"

    def _synthesize_answer(self, query: str, results: List[Dict]) -> str:
        """결과 종합"""
        context = "\n\n".join([
            f"[{r['step']}]\n{r['result']}"
            for r in results
        ])

        prompt = f"""다음 분석 결과를 바탕으로 질문에 답변하세요.

[분석 결과]
{context}

[질문]
{query}

[답변]:"""

        return self.llm_fn(prompt)


# ═══════════════════════════════════════════════════════════════════════════
# 에이전트 팩토리
# ═══════════════════════════════════════════════════════════════════════════

def create_rag_agent(
    search_fn: Callable,
    llm_fn: Callable,
    analyze_fn: Callable,
    clarify_fn: Callable,
    agent_type: str = "basic",
    **kwargs
) -> Any:
    """에이전트 생성 팩토리"""
    if agent_type == "basic":
        return RAGAgent(
            search_fn=search_fn,
            llm_fn=llm_fn,
            analyze_fn=analyze_fn,
            clarify_fn=clarify_fn,
            **kwargs
        )
    elif agent_type == "react":
        tools = {
            "search": lambda q: search_fn(q, 5, None)[1],
            "summarize": lambda t: llm_fn(f"요약: {t[:500]}"),
        }
        return ReActAgent(llm_fn=llm_fn, tools=tools, **kwargs)
    elif agent_type == "plan_execute":
        return PlanAndExecuteAgent(
            llm_fn=llm_fn,
            search_fn=search_fn,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")