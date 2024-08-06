from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

persist_directory = "portfolio_db"
headers_on_split = [("##")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_on_split, strip_headers=False)

def process_file(file_path):
    with open(file_path, "r") as file:
        markdown_text = file.read()
    return markdown_splitter.split_text(markdown_text)

markdown_file_ru = os.path.join(os.path.dirname(__file__), "document_ru.md")
markdown_file_en = os.path.join(os.path.dirname(__file__), "document_en.md")

md_header_splits_ru = process_file(markdown_file_ru)
md_header_splits_en = process_file(markdown_file_en)

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

docs = md_header_splits_en + md_header_splits_ru
existing_ids = set(db.get()["ids"])

for i, doc in enumerate(docs):
    doc_id = f"doc_{i}"
    if doc_id in existing_ids:
        db.update_document(doc_id, doc)
    else:
        db.add_documents([doc], ids=[doc_id])

db.persist()