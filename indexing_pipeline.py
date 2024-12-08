from haystack_integrations.document_stores.weaviate.document_store import WeaviateDocumentStore
from haystack_integrations.components.retrievers.weaviate.embedding_retriever import WeaviateEmbeddingRetriever
from haystack import Pipeline
from haystack.components.routers import FileTypeRouter
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

class IndexingPipeline:
    def __init__(self, url: str = "http://localhost:8080") -> None:
        self.document_store = WeaviateDocumentStore(url = url)

        self.retriever = WeaviateEmbeddingRetriever(document_store=self.document_store)

        self.pipeline = Pipeline()
    
    def build_indexing_pipeline(self):
        self.pipeline.add_component("file_type_router", FileTypeRouter(mime_types=["text/plain"]))

        self.pipeline.add_component("TextConverter", TextFileToDocument())

        preprocessor = DocumentCleaner()

        self.pipeline.add_component("preprocessor", preprocessor)

        embedder = SentenceTransformersDocumentEmbedder()

        self.pipeline.add_component('document_embedder', embedder)

        self.pipeline.add_component("writer", DocumentWriter(self.document_store))

        # Connect all the components
        self.pipeline.connect("file_type_router.text/plain", "TextConverter")
        self.pipeline.connect("TextConverter", "preprocessor.documents")
        self.pipeline.connect("preprocessor", "document_embedder")
        self.pipeline.connect("document_embedder", "writer")
                              
    def run_indexing_pipeline(self, file_paths: list):
        self.pipeline.run(({"file_type_router": {"sources": file_paths}}))
