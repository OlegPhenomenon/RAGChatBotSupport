from groq import Groq
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

TOKEN_LIMIT = 4096

app = FastAPI()

client = Groq(
  api_key=os.getenv("GROQ_API_KEY")
)

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
my_activeloop_org_id = "learningprocess123"
my_activeloop_dataset_name = "my_dataset"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

class ChatInput(BaseModel):
  message: str

class ChatResponse(BaseModel):
  response: str

dialog_history = []

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_input: ChatInput):
  user_input = chat_input.message
  docs = db.similarity_search(user_input, k=3)

  if len(docs) == 0:
    return ChatResponse(response="I'm sorry, I don't have that information. But you can send a message to the email in the contact form.")
  
  template = """
    You are an exceptional customer support chatbot that gently answers questions.
    You know the following context information.
    {chunks_formatted}

    Answer the following question from a customer. Use only information from the previous context information. Do not invent stuff.
    If user asks a question that is not in the context information, answer with "I'm sorry, I don't have that information. But you can send message to email in contact form.".
    If user ask in Russian, you can answer in Russian. If user ask in English, you can answer in English.

    Question: {query}
    Answer:
  """

  chunks_formatted = "\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(docs)])
  message = template.format(chunks_formatted=chunks_formatted, query=user_input)

  dialog_history.append({
    'role': 'user',
    'content': message
  })

  models = ["gemma-7b-it", "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
  chat_completion = client.chat.completions.create(
    messages=dialog_history,
    model=models[1]
  )

  response = chat_completion.choices[0].message.content
  
  dialog_history.append({
    'role': 'assistant',
    'content': response
  })

  if sum(len(message['content']) for message in dialog_history) > TOKEN_LIMIT:
    while sum(len(message['content']) for message in dialog_history) > TOKEN_LIMIT:
      dialog_history.pop(0)

  return ChatResponse(response=response)

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
