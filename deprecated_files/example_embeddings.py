# A very basic example file showing how to get embedding from SAIA cloud

import os
from openai import OpenAI
from dotenv import load_dotenv

# API configuration
api_key = 'f151209de526ea501caa1b52358ef815' # Replace with your API key
base_url = "https://chat-ai.academiccloud.de/v1"
model = "e5-mistral-7b-instruct" # Choose any available model
 
# Start OpenAI client
client = OpenAI(
    api_key = api_key,
    base_url = base_url
)

# Get Embedding
embedding = client.embeddings.create(
    input="Hello, how are you?",
    model = model)

print(embedding)