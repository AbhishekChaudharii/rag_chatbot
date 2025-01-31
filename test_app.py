import pytest
from RAG_Chatbot.app import app  
from embed_store import query_reterival_pipeline

@pytest.fixture
def client():
    """Flask test client fixture."""
    app.testing = True  # Enable testing mode
    with app.test_client() as client:
        yield client

@pytest.fixture
def querying_pipeline():
    """Fixture to initialize the querying pipeline."""
    return query_reterival_pipeline()  # Initialize retrieval pipeline

def test_embedding_and_retrieval(querying_pipeline):
    """Test embedding and retrieval using sample queries."""
    sample_queries = [
        "What is autism?",
        "Explain artificial intelligence.",
        "Who invented the computer?"
    ]
    
    for query in sample_queries:
        results = querying_pipeline.run({
            "query_embedder": {"text": query},
            "prompt_builder": {"query": query},
            "llm": {"generation_kwargs": {"max_new_tokens": 350}},
        })

        assert "llm" in results
        assert "replies" in results["llm"]
        assert len(results["llm"]["replies"]) > 0
        assert isinstance(results["llm"]["replies"][0], str)  # Response should be a string

def test_chat_endpoint(client):
    """Test the chat endpoint with a sample query."""
    response = client.post('/chat', data={'query': 'What is AI?'})
    assert response.status_code == 200
    assert 'response' in response.json
    assert isinstance(response.json['response'], str)  # Chatbot response should be a string

def test_chat_history(client):
    """Test if chat history is stored and retrieved correctly."""
    # Send a chat message
    client.post('/chat', data={'query': 'Hello'})

    # Fetch chat history
    response = client.get('/history')
    assert response.status_code == 200
    assert isinstance(response.json, list)
    assert len(response.json) > 0  # At least one chat should be stored

def test_empty_chat_request(client):
    """Test if the app handles empty input gracefully."""
    response = client.post('/chat', data={'query': ''})
    assert response.status_code == 400  # Bad request for empty input

def test_invalid_endpoint(client):
    """Test if accessing an invalid endpoint returns 404."""
    response = client.get('/invalid_endpoint')
    assert response.status_code == 404  # Page not found
