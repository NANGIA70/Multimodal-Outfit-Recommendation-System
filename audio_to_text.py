import requests
import os

API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"

token = "Bearer " + os.getenv('HF_API_TOKEN')
headers = {"Authorization": token}

print(headers)
def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()