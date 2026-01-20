"""
RAG 패키지 - 리팩토링 v5.1
- 검색 품질 개선 (confidence, threshold)
- 청크 크기 최적화 (300자)
- 블록 기반 청킹 정상 작동
"""

from .chunker import (
    create_chunks,
    create_chunks_from_blocks,
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
    search_advanced,  # 새로 추가
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
    # 새로 추가된 상수
    DEFAULT_SIMILARITY_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    SearchResult,
    SearchResponse,
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
)

from .parser import (
    ParsedDocument,
    ContentBlock,
    parse_plain_text,
    parse_articles,
)

from .prompt import (
    build_rag_prompt,
    build_chunk_prompt,
    build_summary_prompt,
)

__version__ = "5.1.0"
