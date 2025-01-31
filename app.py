from flask import Flask, request, render_template
import os
from dotenv import load_dotenv
import mysql.connector
import embed_store
import data_preprocessing

app = Flask(__name__)

# Load environment variables
load_dotenv()
my_sql_username = os.getenv("MYSQL_USER")
my_sql_password = os.getenv("MYSQL_PASSWORD")

# Initialize Database & Query Pipeline (Run Once)
data_preprocessing.pipeline_building()
data_preprocessing.initialize_db(my_sql_username, my_sql_password)
querying_pipeline = embed_store.query_reterival_pipeline()

# Function to Store Chat History
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

# Function to Retrieve Chat History
def get_chat_history():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user=my_sql_username,
            password=my_sql_password,
            database="RAG_CHAT",
        )
        cursor = connection.cursor()
        cursor.execute("SELECT timestamp, user_query, result FROM chat_history ORDER BY timestamp DESC")
        history = cursor.fetchall()
        return history
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Homepage: Chat Interface
@app.route("/chat", methods=["GET", "POST"])
def chat():
    response = None
    if request.method == "POST":
        user_query = request.form["query"]
        results = querying_pipeline.run({
            "query_embedder": {"text": user_query},
            "prompt_builder": {"query": user_query},
            "llm": {"generation_kwargs": {"max_new_tokens": 350}},
        })
        response = results["llm"]["replies"][0]
        store_chat(user_query, response)  # Save chat to database

    return render_template("chat.html", response=response)

# Chat History Page
@app.route("/history", methods=["GET"])
def history():
    chat_history = get_chat_history()
    return render_template("history.html", chat_history=chat_history)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
