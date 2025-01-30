import mysql.connector
from pathlib import Path
from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentSplitter,DocumentCleaner
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder




def pipeline_building():
    """
    Build and run the data preprocessing pipeline if data does not already exist in the document store.
    """
    # Check if the document store already has data
    document_store = ChromaDocumentStore(persist_path="./chroma_db")
    if document_store.count_documents() > 0:
        print(document_store.count_documents())
        print("Data already exists in the document store. Skipping data processing.")
        return document_store

    file_paths = [Path("data/autism.txt")] 

    document_embedder = HuggingFaceAPIDocumentEmbedder(api_type="serverless_inference_api",
                                                            api_params={"model": "BAAI/bge-small-en-v1.5"})

    # Persistent Storage
    document_store = ChromaDocumentStore(persist_path="./chroma_db")

    indexing = Pipeline()
    indexing.add_component("converter", TextFileToDocument()) 
    indexing.add_component("cleaner", DocumentCleaner())
    indexing.add_component(instance=DocumentSplitter(split_by="word", split_length=200), name="splitter")
    indexing.add_component("document_embedder", document_embedder)
    indexing.add_component("writer", DocumentWriter(document_store))


    # Connect the components
    indexing.connect("converter", "cleaner")
    indexing.connect("cleaner", "splitter")
    indexing.connect("splitter", "document_embedder")
    indexing.connect("document_embedder", "writer")  


    indexing.run({"converter": {"sources": file_paths}})
    return document_store

def initialize_db(my_sql_username,my_sql_password):
    mysql_config = {
    "host": "localhost",
    "user": my_sql_username,  # Replace with your MySQL username
    "password": my_sql_password,  # Replace with your MySQL password
    }
    try:
        # Connect to MySQL server
        connection = mysql.connector.connect(**mysql_config)
        cursor = connection.cursor()

        # Create database if it doesn't exist
        cursor.execute("CREATE DATABASE IF NOT EXISTS RAG_CHAT")
        cursor.execute("USE RAG_CHAT")

        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_query TEXT NOT NULL,
            result TEXT NOT NULL
            );
        """)

        print("Database and table initialized successfully.")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()



# if __name__=="__main__":
#     # Load environment variables
#     load_dotenv()
#     hf_token = os.getenv("HF_API_TOKEN")
#     print("Hugging Face Token Loaded:", hf_token[:5] + "..." if hf_token else "Not Found")
#     data_storing_pipeline = pipeline_building()
