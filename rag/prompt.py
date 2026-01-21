"""
RAG 프롬프트 템플릿 v6.0
"""


def build_rag_prompt(query: str, context: str, language: str = "ko") -> str:
    """RAG 프롬프트 생성"""
    if language == "ko":
        return f"""당신은 규정(SOP) 전문가입니다. 아래 [참고 문서]를 바탕으로 사용자의 질문에 답변하세요.

지침:
- 문서에 없는 내용은 답변에 포함하지 마세요.
- 추측이나 외부 지식을 사용하지 마세요.
- 근거가 되는 조항(예: 제N조)이 있다면 반드시 언급하세요.
- 정보를 찾을 수 없다면 '해당 문서에서 정보를 찾을 수 없습니다.'라고 답변하세요.

[참고 문서]
{context}

[사용자 질문]
{query}

[전문가 답변]:"""
    else:
        return f"""You are an expert in regulations and SOPs. Answer based ONLY on the provided documents.

Instructions:
- Do not include information not in the documents.
- Cite specific articles (e.g., Article N) when relevant.
- If not found, say "Information not available in provided documents."

[Reference Documents]
{context}

[User Question]
{query}

[Expert Answer]:"""


def build_chunk_prompt(query: str, chunk_text: str, language: str = "ko") -> str:
    """단일 청크 기반 프롬프트"""
    if language == "ko":
        return f"""아래 [문서 조각]을 바탕으로 질문에 답변하세요.

지침:
- 문서 조각에 없는 내용은 답변에 포함하지 마세요.
- 정보를 찾을 수 없다면 '해당 내용에서 정보를 찾을 수 없습니다.'라고 답변하세요.

[문서 조각]
{chunk_text}

[질문]
{query}

[답변]:"""
    else:
        return f"""Answer based ONLY on the following document chunk.

[Document Chunk]
{chunk_text}

[Question]
{query}

[Answer]:"""


def build_summary_prompt(text: str, language: str = "ko") -> str:
    """요약 프롬프트"""
    if language == "ko":
        return f"""다음 문서의 핵심 내용을 요약해주세요.

[문서]
{text}

[요약]:"""
    else:
        return f"""Summarize the key points of this document.

[Document]
{text}

[Summary]:"""


def build_clarification_prompt(query: str, options: list, language: str = "ko") -> str:
    """되묻기 프롬프트"""
    options_text = "\n".join([f"- {opt}" for opt in options])

    if language == "ko":
        return f"""사용자가 "{query}"에 대해 질문했습니다.
다음 문서들이 검색되었습니다:
{options_text}

어떤 문서를 바탕으로 답변할지 정중하게 물어보세요.
한국어로 짧고 명확하게 응답하세요."""
    else:
        return f"""The user asked about "{query}".
Found documents:
{options_text}

Politely ask which document to reference.
Keep your response short and clear."""