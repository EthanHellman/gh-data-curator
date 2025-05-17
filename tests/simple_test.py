import os
import requests

api_key = os.environ.get("OPENAI_API_KEY")
print(f"API key found: {bool(api_key)}")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello!"}
    ]
}

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers=headers,
    json=payload
)

print(f"Status code: {response.status_code}")
print(f"Response: {response.json() if response.status_code == 200 else response.text}")