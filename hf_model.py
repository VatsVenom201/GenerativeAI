import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="deepseek-ai/DeepSeek-V3.2",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)


# Store conversation history
messages = []

while True:
    user_query = input("You: ")

    if user_query.lower() == "exit":
        break

    # Add user message to history
    messages.append({"role": "user", "content": user_query})

    # Send full history to model
    response = client.chat_completion(
        messages=messages,
        max_tokens=512
    )

    assistant_reply = response.choices[0].message["content"]

    print("Bot:", assistant_reply)

    # Add assistant reply to history
    messages.append({"role": "assistant", "content": assistant_reply})