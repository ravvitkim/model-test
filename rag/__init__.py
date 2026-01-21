"""
RAG 패키지 v6.0
- Docling 기반 문서 파싱 (표 지원)
- 개선된 검색 (similarity_threshold)
- 가독성 개선된 메타데이터 (제N조 형식)
"""

from .document_loader import (
    load_document,
    get_supported_extensions,
    ParsedDocument,
    ContentBlock,
)

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
    search_advanced,
    add_documents,
    add_single_text,
    list_documents,
    list_collections,
    delete_by_doc_name,
    delete_all,
    get_embedding_model_info,
    filter_compatible_models,
    is_model_compatible,
    get_collection_info,
    EMBEDDING_MODEL_SPECS,
    MAX_EMBEDDING_DIM,
    MAX_MEMORY_MB,
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

from .prompt import (
    build_rag_prompt,
    build_chunk_prompt,
    build_summary_prompt,
    build_clarification_prompt,
)

__version__ = "6.0.0"