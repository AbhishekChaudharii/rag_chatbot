from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.components.embedders import HuggingFaceAPITextEmbedder
from haystack.components.builders import PromptBuilder
from haystack import Pipeline

def query_reterival_pipeline():
    """Returns pipeline for querying"""

    

    document_store = ChromaDocumentStore(persist_path="./chroma_db")

    prompt = """
    Answer the query based on the provided context.
    If the context does not contain the answer, say 'Answer not found'.
    Context:
    {% for doc in documents %}
    {{ doc.content }}
    {% endfor %}
    query: {{query}}
    Answer:
    """
    prompt_builder = PromptBuilder(template=prompt)

    llm = HuggingFaceAPIGenerator(api_type="serverless_inference_api",
                                api_params={"model": "HuggingFaceH4/zephyr-7b-beta"})

    retriever = ChromaEmbeddingRetriever(document_store)
    query_embedder = HuggingFaceAPITextEmbedder(api_type="serverless_inference_api",
                                            api_params={"model": "BAAI/bge-small-en-v1.5"})
    querying = Pipeline()
    querying.add_component("query_embedder", query_embedder)
    querying.add_component("retriever", retriever)
    querying.add_component("prompt_builder", prompt_builder)
    querying.add_component("llm", llm)

    querying.connect("query_embedder.embedding", "retriever.query_embedding")
    querying.connect("retriever.documents", "prompt_builder.documents")
    querying.connect("prompt_builder", "llm")

    return querying

    
  

# if __name__=="__main__":
#     # load_dotenv()
#     # hf_token = os.getenv("HF_API_TOKEN")
#     # print("Hugging Face Token Loaded:", hf_token[:5] + "..." if hf_token else "Not Found")
#     # document_store = data_preprocessing.pipeline_building()
#     # print(query_reterival_pipeline(document_store=document_store,query="Is adhd real?"))


