#!/usr/bin/env python3

from llama_index.core.query_pipeline import InputComponent
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.cli.rag.base import QueryPipelineQueryEngine
# from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import Settings
from llama_index.cli.rag import RagCLI
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.chat_engine import CondenseQuestionChatEngine

from common import load_vector_store_index


EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
LLM_MODEL_NAME = "llama3.1"

Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
Settings.llm = Ollama(model=LLM_MODEL_NAME, context_window=8000)
# Settings.embed_model = OpenAIEmbedding()
# Settings.llm = OpenAI(model="gpt-3.5-turbo")

index = load_vector_store_index(["data", "dp-ilqr"], Settings.embed_model)
ingestion_pipeline = IngestionPipeline(
    transformations=[Settings.embed_model],
    vector_store=index.vector_store,  # your vector store instance
    # docstore=docstore,
    # cache=IngestionCache(),
)

# Setting up the custom QueryPipeline is optional!
# You can still customize the vector store, LLM, and ingestion transformations without
# having to customize the QueryPipeline
query_pipeline = QueryPipeline()
# query_pipeline.add_modules(...)
# query_pipeline.add_link(...)

# Add a root module to the pipeline
query_pipeline.add_modules(dict(
    input=InputComponent(),
    query=index.as_query_engine(),
))
query_pipeline.add_link("input", "query")
# query_pipeline.set_root("input")

query_engine = QueryPipelineQueryEngine(query_pipeline=query_pipeline)  # type: ignore
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine, llm=Settings.llm
)

# To get this to work, we need to make the following change:
# env/lib/python3.12/site-packages/llama_index/cli/rag/base.py @ line 324
# -       chat_engine.streaming_chat_repl()
# +       chat_engine.chat_repl()
rag_cli_instance = RagCLI(
    ingestion_pipeline=ingestion_pipeline,
    llm=Settings.llm,  # optional
    query_pipeline=query_pipeline,  # optional
    chat_engine=chat_engine,
)

if __name__ == "__main__":
    rag_cli_instance.cli()
