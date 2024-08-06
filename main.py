from groq import Groq
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging
import asyncio

load_dotenv()

TOKEN_LIMIT = 4096

app = FastAPI()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
persist_directory = "portfolio_db"
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

class ChatInput(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

dialog_history = []

async def call_groq_api(messages, model):
    try:
        chat_completion = await asyncio.to_thread(
            client.chat.completions.create,
            messages=messages,
            model=model
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling Groq API: {e}")
        raise HTTPException(status_code=500, detail="Error processing request")

async def process_chat(chat_input: ChatInput):
    user_input = chat_input.message
    docs = db.similarity_search(user_input, k=3)

    logger.info(f"Found {len(docs)} documents")

    if len(docs) == 0:
        return "I'm sorry, I don't have that information. But you can send a message to the email in the contact form."
    
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

    manager_template = """
    You are an exceptional chatbot manager. Your task is to review the text provided by your subordinates. You need to check the grammar and content and correct it to a relevant version.

    The happiness of our customers depends on the quality of the text, so make sure the text is clear and does not mix languages. If this happens, you should correct the text to convey the message to the customer clearly.

    The text can be either in Russian or in English. If user asks in Russian, you must answer in Russian. If user asks in English, you must answer in English.
    We are the technical support for the user, so it is important to communicate with them in a polite tone, with humor, and to provide comprehensive answers to their questions.

    Please answer directly to the user's question. Do not invent stuff. Don't tell that you changed the text. Just provide the corrected text.
    DON'T USE "Here is the corrected text:" or any other similar phrases. Use only direct text.

    Let's think step by step. First, let's check the text for grammar and content. If you see any mistakes, correct them. If the text is correct, you can leave it as it is.
    If the text mixes languages, correct it to a relevant version. If you see that a foreign word has been added to a message in Russian, replace it with a Russian word.
    If you see that a foreign word has been added to a message in English, replace it with an English word.

    You should not mention your actions or what you did, just send the corrected version without any explanations.

    Subordinate's text: {text}

    Corrected text:
    """

    chunks_formatted = "\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(docs)])
    message = template.format(chunks_formatted=chunks_formatted, query=user_input)
    models = ["gemma-7b-it", "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "llama-3.1-405b-reasoning"]

    dialog_history.append({
        'role': 'user',
        'content': message
    })

    logger.info(f"Send message {message}")

    response = await call_groq_api(dialog_history, models[1])

    logger.info(f"Received response {response}")

    logger.info(f"Send message to critic: {response}")
    manager_message = manager_template.format(text=response)
    dialog_history.append({
        'role': 'user',
        'content': manager_message
    })

    response = await call_groq_api(dialog_history, models[1])

    logger.info(f"Received response {response}")
    
    dialog_history.append({
        'role': 'assistant',
        'content': response
    })

    if sum(len(message['content']) for message in dialog_history) > TOKEN_LIMIT:
        while sum(len(message['content']) for message in dialog_history) > TOKEN_LIMIT:
            dialog_history.pop(0)

    return response

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_input: ChatInput, background_tasks: BackgroundTasks):
    try:
        task = asyncio.create_task(process_chat(chat_input))
        background_tasks.add_task(task)
        response = await asyncio.wait_for(task, timeout=120.0)  # Увеличен таймаут до 2 минут
        return ChatResponse(response=response)
    except asyncio.TimeoutError:
        logger.error("Request timed out")
        return ChatResponse(response="I'm sorry, but the request is taking longer than expected. Please try again later.")
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=120)