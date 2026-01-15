"""
RAG 프롬프트 템플릿
"""


def build_rag_prompt(query: str, context: str, language: str = "ko") -> str:
    """
    전체 컨텍스트 기반 RAG 프롬프트 생성
    """
    if language == "ko":
        prompt = f"""당신은 제공된 문서를 바탕으로 질문에 답하는 AI 비서입니다.
아래 지침을 반드시 따르세요:
1. 반드시 [문서 내용]에 있는 정보만을 사용하여 답변하십시오.
2. 문서 내용으로 질문에 답할 수 없는 경우, 절대 추측하지 말고 '제공된 문서에서 관련 정보를 찾을 수 없습니다.'라고 답변하십시오.
3. 답변은 친절하고 정중한 문체로 작성하십시오.

[문서 내용]
{context}

[질문]
{query}

[답변]
"""
    else:
        prompt = f"""You are an AI assistant that answers questions based on the provided context.
Please follow these instructions:
1. Answer the question ONLY using the information in [Context].
2. If the context does not contain the answer, do NOT guess. Instead, say "The provided context does not contain relevant information."
3. Be concise and accurate.

[Context]
{context}

[Question]
{query}

[Answer]
"""
    return prompt


def build_chunk_prompt(query: str, chunk_text: str, language: str = "ko") -> str:
    """
    개별 청크 기반 프롬프트 생성
    """
    if language == "ko":
        prompt = f"""당신은 아래의 [문서 조각]만을 참고하여 질문에 답해야 합니다.
지침:
- 문서 조각에 없는 내용은 절대 답변에 포함하지 마십시오.
- 추측이나 외부 지식을 사용하지 마십시오.
- 정보를 찾을 수 없다면 '해당 문장에서는 정보를 찾을 수 없습니다.'라고 답변하십시오.

[문서 조각]
{chunk_text}

[질문]
{query}

[답변]
"""
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

[Answer]
"""
    return prompt