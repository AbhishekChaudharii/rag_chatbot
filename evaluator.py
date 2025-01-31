import json
import os
import numpy as np
from dotenv import load_dotenv
import mysql.connector
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from embed_store import query_reterival_pipeline
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import process

# Load environment variables
load_dotenv()
my_sql_username = os.getenv("MYSQL_USER")
my_sql_password = os.getenv("MYSQL_PASSWORD")

# Load prebuilt query pipeline
querying_pipeline = query_reterival_pipeline()

# Load reference answers (Ground Truth)
with open("data/ground_truth.json", "r") as f:
    ground_truth = json.load(f)

# Initialize Sentence Transformer for embedding similarity
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Function to retrieve stored chat history
def get_chat_history():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user=my_sql_username,
            password=my_sql_password,
            database="RAG_CHAT",
        )
        cursor = connection.cursor()
        cursor.execute("SELECT user_query, result FROM chat_history")
        history = cursor.fetchall()
        return history  
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def evaluate_retrieval():
    document_store = ChromaDocumentStore(persist_path="./chroma_db")
    correct_retrievals = 0
    total_queries = len(ground_truth)

    for query, expected_docs in ground_truth.items():
        results = querying_pipeline.run(
            {
                "query_embedder": {"text": query},
                "prompt_builder": {"query": query},
                "llm": {"generation_kwargs": {"max_new_tokens": 350}},
            }
        )

     
        retrieved_docs = results['llm']['replies']


        # Calculate similarity using embeddings
        retrieved_embeddings = embedding_model.encode(retrieved_docs)
        expected_embeddings = embedding_model.encode(expected_docs)

        # Compute cosine similarity
        similarities = cosine_similarity(retrieved_embeddings, expected_embeddings)
        if np.max(similarities) > 0.8:  # Threshold for a correct retrieval
            correct_retrievals += 1

    return correct_retrievals / total_queries if total_queries > 0 else 0


# Function to evaluate response quality
# def evaluate_responses():
#     history = get_chat_history()
    
#     rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    
#     bleu_scores, rouge_scores = [], []

#     for user_query, generated_response in history:

#         if user_query in ground_truth:
#             expected_response = ground_truth[user_query][0]
            
#             # Compute BLEU Score
#             bleu = sentence_bleu([expected_response.split()], generated_response.split())
#             bleu_scores.append(bleu)

#             # Compute ROUGE Score
#             rouge_score = rouge.score(expected_response, generated_response)
#             rouge_scores.append(rouge_score["rougeL"].fmeasure)

#     return {
#         "BLEU Score": np.mean(bleu_scores),
#         "ROUGE Score": np.mean(rouge_scores),
#     }

def find_closest_query(user_query, ground_truth):
    best_match, score = process.extractOne(user_query, list(ground_truth.keys()))
    return best_match if score > 70 else None  # Only consider matches above 70% similarity

# Function to evaluate response quality
def evaluate_responses():
    history = get_chat_history()  # Fetch chat history from the database
    
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    smoothing = SmoothingFunction().method1  # Use smoothing for BLEU scores

    bleu_scores, rouge_scores = [], []

    for user_query, generated_response in history:
        closest_query = find_closest_query(user_query, ground_truth)
        if not closest_query:
            continue  # Skip if no close match is found
        
        expected_response = ground_truth[closest_query][0]  # Get the expected answer

        # Compute BLEU Score with smoothing
        bleu = sentence_bleu(
            [expected_response.split()], 
            generated_response.split(),
            smoothing_function=smoothing
        )
        bleu_scores.append(bleu)

        # Compute ROUGE Score
        rouge_score = rouge.score(expected_response, generated_response)
        rouge_scores.append(rouge_score["rougeL"].fmeasure)

    return {
        "BLEU Score": np.mean(bleu_scores) if bleu_scores else 0.0,
        "ROUGE Score": np.mean(rouge_scores) if rouge_scores else 0.0,
    }

if __name__ == "__main__":
    # print("Evaluating Retrieval...")
    # retrieval_accuracy = evaluate_retrieval()
    # print(f"Retrieval Accuracy: {retrieval_accuracy:.2f}")

    print("\nEvaluating Responses...")
    response_scores = evaluate_responses()
    print(f"BLEU Score: {response_scores['BLEU Score']:.2f}")
    print(f"ROUGE Score: {response_scores['ROUGE Score']:.2f}")
