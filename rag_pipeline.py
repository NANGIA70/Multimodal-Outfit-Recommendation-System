import os
from haystack import Pipeline
from haystack.utils import Secret
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.weaviate.document_store import WeaviateDocumentStore
from haystack_integrations.components.retrievers.weaviate.embedding_retriever import WeaviateEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceAPIGenerator


class RagPipeline:
    def __init__(self, url: str = "http://localhost:8080") -> None:
        self.document_store = self.get_document_store(url)
        self.retriever =  self.get_retriever()
        self.embedder = self.get_query_embedder()
        self.prompt_builder = self.get_prompt()
        self.llm_model = self.get_llm_model()
        self.pipeline = Pipeline()

    def get_document_store(self, url):
        document_store =  WeaviateDocumentStore(url = url)
        return document_store
    
    def get_retriever(self):
        retriever =  WeaviateEmbeddingRetriever(document_store=self.document_store)
        return retriever

    def get_prompt(self):
        template = """
        Given the following information, answer the question.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """

        prompt_builder = PromptBuilder(template=template)

        return prompt_builder 

    def get_llm_model(self):
        llm_model = HuggingFaceAPIGenerator(api_type="serverless_inference_api",
                                        api_params={"model": "mistralai/Mistral-7B-Instruct-v0.3", "stop": ["\n\n"]},
                                        token= Secret.from_env_var("HF_API_TOKEN"))

        return llm_model
    
    def get_query_embedder(self):
        embedder = SentenceTransformersTextEmbedder()

        return embedder 
    
    def build_rag_pipeline(self):
        self.pipeline.add_component("text_embedder", self.embedder)
        self.pipeline.add_component("retriever", self.retriever)
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.llm_model)

        self.pipeline.connect("text_embedder", "retriever")
        self.pipeline.connect("retriever", "prompt_builder")
        self.pipeline.connect("prompt_builder", "llm")

    def run_rag_pipeline(self, query: str):
        response = self.pipeline.run({"text_embedder": {"text": query}, "prompt_builder": {"question": query}})

        return response["llm"]["replies"][0]