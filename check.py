import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Available Models:")
for model in client.models.list():
    print(f"- {model.name}")

response = client.models.generate_content(
    model="models/gemini-2.5-flash", 
    contents="Verify the API is working. Reply with 'Success!'"
)
print(f"\nTest Response: {response.text}")