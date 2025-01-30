from flask import Flask, request, render_template
import os
from dotenv import load_dotenv
import data_preprocessing
import embed_store
import mysql.connector

app = Flask(__name__)

# Load environment variables
load_dotenv()
my_sql_username = os.getenv("MYSQL_USER")
my_sql_password = os.getenv("MYSQL_PASSWORD")

# Initialize document store and query pipeline only once
data_preprocessing.initialize_db(my_sql_username, my_sql_password)
document_store = data_preprocessing.pipeline_building()
querying_pipeline = embed_store.query_reterival_pipeline()

# Function to store chat history in MySQL
def store_chat(user_query, result):
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user=my_sql_username,
            password=my_sql_password,
            database="RAG_CHAT",
        )
        cursor = connection.cursor()
        query = "INSERT INTO chat_history (user_query, result) VALUES (%s, %s)"
        cursor.execute(query, (user_query, result))
        connection.commit()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

@app.route("/", methods=["GET", "POST"])
def index():
    response_text = None
    user_query = None

    if request.method == "POST":
        user_query = request.form["query"]
        results = querying_pipeline.run(
            {
                "query_embedder": {"text": user_query},
                "prompt_builder": {"query": user_query},
                "llm": {"generation_kwargs": {"max_new_tokens": 350}},
            }
        )
        response_text = results["llm"]["replies"][0]
        store_chat(user_query, response_text)

    return render_template("index.html", user_query=user_query, response_text=response_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
