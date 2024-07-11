import httpx
import json
import time
import random

def rerank_request(client):
    documents = [
        "This is a document not related to the python package infinity_emb, hence...",
        "Paris is in France!",
        "infinity_emb is a package for sentence embeddings and rerankings using transformer models in Python!"
    ]  # Repeat to simulate a large number of documents

    headers = {
        "Content-Type": "application/json",
        "Authorization": "",
    }
    data = {"model":"michaelfeil/bge-small-en-v1.5","input":documents}

    response = client.post('http://localhost:7997/embeddings', headers=headers, json=data)
    print(response)

def main():
    with httpx.Client() as client:
        # while True:
        rerank_request(client)
        # Simulate wait time between requests (1 to 5 seconds)
        time.sleep(random.uniform(1, 5))

if __name__ == "__main__":
    main()