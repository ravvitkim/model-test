"""
RAG 프롬프트 템플릿
"""


def build_rag_prompt(query: str, context: str, language: str = "ko") -> str:
    """
    RAG 프롬프트 생성
    
    Args:
        query: 사용자 질문
        context: 검색된 문서 컨텍스트
        language: "ko" 또는 "en"
    """
    if language == "ko":
        prompt = f"""당신은 규정(SOP) 전문가입니다. 아래 제공된 [참고 문서]의 내용을 바탕으로 사용자의 질문에 답변하세요.

지침:
- 문서에 없는 내용은 절대 답변에 포함하지 마십시오.
- 추측이나 외부 지식을 사용하지 마십시오.
- 답변 시 근거가 되는 조항(예: 제N조)이 있다면 반드시 언급하세요.
- 정보를 찾을 수 없다면 '해당 문서에서는 정보를 찾을 수 없습니다.'라고 답변하십시오.

[참고 문서]
{context}

[사용자 질문]
{query}

[전문가 답변]:"""
    else:
        prompt = f"""You are an expert in regulations and SOPs. Answer the user's question based ONLY on the provided [Reference Documents].

Instructions:
- Do not include any information not present in the reference documents.
- Do not use external knowledge or guesses.
- If citing specific articles (e.g., Article N), mention them in your answer.
- If the information cannot be found, say "The information is not available in the provided documents."

[Reference Documents]
{context}

[User Question]
{query}

[Expert Answer]:"""
    
    return prompt


def build_chunk_prompt(query: str, chunk_text: str, language: str = "ko") -> str:
    """
    단일 청크 기반 프롬프트 생성
    
    Args:
        query: 사용자 질문
        chunk_text: 단일 청크 텍스트
        language: "ko" 또는 "en"
    """
    if language == "ko":
        prompt = f"""아래 제공된 [문서 조각]의 내용을 바탕으로 질문에 답변하십시오.

지침:
- 문서 조각에 없는 내용은 절대 답변에 포함하지 마십시오.
- 추측이나 외부 지식을 사용하지 마십시오.
- 정보를 찾을 수 없다면 '해당 문장에서는 정보를 찾을 수 없습니다.'라고 답변하십시오.

[문서 조각]
{chunk_text}

[질문]
{query}

[답변]:"""
    else:
        prompt = f"""Answer the question based ONLY on the following [Document Chunk].

Instructions:
- Do not include any information not present in the document chunk.
- Do not use external knowledge or guesses.
- If you cannot answer, say "Information not found in this chunk."

[Document Chunk]
{chunk_text}

[Question]
{query}

[Answer]:"""
    
    return prompt


def build_summary_prompt(text: str, language: str = "ko") -> str:
    """
    요약 프롬프트 생성
    """
    if language == "ko":
        prompt = f"""다음 문서를 핵심 내용 위주로 요약해주세요.

[문서]
{text}

[요약]:"""
    else:
        prompt = f"""Summarize the following document, focusing on key points.

[Document]
{text}

[Summary]:"""
    
    return prompt


def build_clarification_prompt(query: str, options: list, language: str = "ko") -> str:
    """
    되묻기 프롬프트 생성
    """
    options_text = "\n".join([f"- {opt}" for opt in options])
    
    if language == "ko":
        prompt = f"""사용자가 "{query}"에 대해 질문했습니다.
관련하여 다음 문서들이 검색되었습니다:
{options_text}

사용자에게 어떤 문서의 내용을 바탕으로 답변을 드릴지 정중하게 물어보세요.
답변은 반드시 한국어로 짧고 명확하게 하세요."""
    else:
        prompt = f"""The user asked about "{query}".
The following documents were found:
{options_text}

Politely ask the user which document they would like to reference for the answer.
Keep your response short and clear."""
    
    return prompt