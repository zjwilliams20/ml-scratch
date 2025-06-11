#!/usr/bin/env python

from pathlib import Path

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
LLM_MODEL_NAME = "llama3.1"



# Settings control global defaults
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
Settings.llm = Ollama(
    model=LLM_MODEL_NAME,
    request_timeout=360.0,
    # Manually set the context window to limit memory usage
    context_window=8000,
)

if not STORAGE_DIR.is_dir():
    STORAGE_DIR.mkdir(parents=True)
    # Create a RAG tool using LlamaIndex
    index = generate_vector_store_index(["data", "dp-ilqr"], Settings.embed_model)
    # Save the index
    index.storage_context.persist(STORAGE_DIR)
else:
    # Later, load the index
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(
        storage_context,
        # we can optionally override the embed_model here
        # it's important to use the same embed_model as the one used to build the index
        embed_model=Settings.embed_model,
    )

query_engine = index.as_query_engine(
    # we can optionally override the llm here
    llm=Settings.llm,
)
response = query_engine.query("What are some examples of dynamical models in dp-ilqr?")
print(response)

