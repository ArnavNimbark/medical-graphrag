import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")

genai.configure(api_key=api_key)

for model in genai.list_models():
    print(model.name)
