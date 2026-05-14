from google import genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Graph context
graph_context = """
Warfarin interacts with Aspirin.
Effect: Bleeding Risk.
"""

# User question
question = "Why is Warfarin dangerous with Aspirin?"

# Prompt
prompt = f"""
You are a medical reasoning assistant.

Using the graph relationship below, explain the drug interaction clearly.

Graph Context:
{graph_context}

Question:
{question}

Provide a concise medical explanation.
"""

# Generate response
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt
)

print("\n=== Gemini Response ===\n")

print(response.text)