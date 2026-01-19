"""
RAG 패키지 - 텍스트 유사도 + RAG + Ollama + 에이전트
v5.0 - 확장 청킹 + 임베딩 모델 필터링
"""

from .chunker import (
    create_chunks,
    get_available_methods,
    CHUNK_METHODS,
    Chunk,
    RecursiveCharacterTextSplitter,
    SemanticSplitter,
    split_by_sentences,
    split_by_paragraphs,
    split_by_articles,
    split_recursive,
    split_semantic,
    split_by_llm,
)

from .vector_store import (
    search,
    search_with_context,
    add_documents,
    add_single_text,
    list_documents,
    list_collections,
    delete_by_doc_name,
    delete_all,
    get_embedding_model_info,
    filter_compatible_models,
    is_model_compatible,
    EMBEDDING_MODEL_SPECS,
    MAX_EMBEDDING_DIM,
    MAX_MEMORY_MB,
)

from .llm import (
    get_llm_response,
    OllamaLLM,
    analyze_search_results,
    generate_clarification_question,
    OLLAMA_MODELS,
    HUGGINGFACE_MODELS,
)

from .document_loader import (
    load_document,
    get_supported_extensions,
    clean_text,
)

from .prompt import (
    build_rag_prompt,
    build_chunk_prompt,
    build_summary_prompt,
)

__version__ = "5.0.0"