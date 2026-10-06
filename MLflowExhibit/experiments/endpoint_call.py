import requests

response = requests.post(
    "http://127.0.0.1:5000/gateway/agentic-ai-endpoint/mlflow/invocations",
    json={
        "messages": [
            {
                "role": "user",
                "content": "What is the balance of account A100?"
            }
        ]
    },
)

print("Response:", response.json())