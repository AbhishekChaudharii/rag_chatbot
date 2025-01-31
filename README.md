# RAG Chatbot

This is a **Flask-based RAG (Retrieval-Augmented Generation) Chatbot** that uses a **MySQL database** for storing chat history. The chatbot retrieves relevant documents and generates responses based on user queries.

## **Features**
✅ Simple **Flask API** for chatting and retrieving chat history

✅ Uses **MySQL** for storing chat history

✅ **Automatic database setup** on the first run

---
## **Installation**

### **1️. Prerequisites**
Ensure you have the following installed on your system:
- Python **3.9+**
- MySQL **8.0+**

### **2️. Clone the Repository**

### **3️. Install Dependencies**
```bash
pip install -r requirements.txt
```
### **4️. Create a huggingface api key**
Token type = read

[Tutorial to create API key](https://huggingface.co/docs/hub/en/security-tokens)

### **5️. Create a .env file**
Enter your hugging face api key in `HF_API_TOKEN`

Your mysql username and password in `MYSQL_USER` and `MYSQL_PASSWORD` respectively 
```bash
HF_API_TOKEN=""
MYSQL_USER=""
MYSQL_PASSWORD=""
```

### **6️. Run the Flask App**
```bash
python app.py
```
To chat: **`http://localhost:5001/chat`**

To view history: **`http://localhost:5001/history`**

---
## **API Endpoints**
### **Chat with the bot**
```bash
curl -X POST -d "query=What is AI?" http://127.0.0.1:5001/chat
```

### **Retrieve chat history**
```bash
curl -X GET http://127.0.0.1:5001/history
```

