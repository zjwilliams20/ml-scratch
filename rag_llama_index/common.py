#!/usr/bin/env python3


from pathlib import Path

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
LLM_MODEL_NAME = "llama3.1"
STORAGE_DIR = Path("storage")

def load_vector_store_index(save_dirs, embed_model):
    """Loads or generates a vector store index from the documents in the specified directories."""
    if not STORAGE_DIR.is_dir():
        # STORAGE_DIR.mkdir(parents=True)
        # Create a RAG tool using LlamaIndex
        documents = []
        for save_dir in save_dirs:
            reader = SimpleDirectoryReader(save_dir)
            documents.extend(reader.load_data())
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
        )
        # Save the index
        index.storage_context.persist(STORAGE_DIR)
    else:
        # Later, load the index
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(
            storage_context,
            # we can optionally override the embed_model here
            # it's important to use the same embed_model as the one used to build the index
            embed_model=embed_model,
        )
    return index