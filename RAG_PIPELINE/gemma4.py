from google import genai
from google.genai import types

# Set up your API key (or use env var)
client = genai.Client(api_key="AIzaSyB7Qv5yjL9G3yzgrEhg2izn4h92zq5j3A4")

# Model you want to use
model_id = "gemma-4-31b-it"

# Read local image
with open("1.png", "rb") as f:
    image_bytes = f.read()

# Create a Part from the raw bytes + MIME type
image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

# Content can be a list of parts/text mixed
contents = [
    # You can put text first or after the image — Gemini API handles multimodal
    "Describe the objects in the image below and give me insights:",
    image_part
]

# Call the API
response = client.models.generate_content(
    model=model_id,
    contents=contents
)

# Print the model's response text
print(response.text)
