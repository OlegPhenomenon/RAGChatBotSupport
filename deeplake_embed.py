from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
import os
from dotenv import load_dotenv

load_dotenv()

# Prepare the file
headers_on_split = [("##")]
mardown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_on_split, strip_headers=False)

markdown_file_ru = os.path.join(os.path.dirname(__file__), "document_ru.md")
with open(markdown_file_ru, "r") as file:
  markdown_text_ru = file.read()
md_header_splits_ru = mardown_splitter.split_text(markdown_text_ru)

markdown_file_en = os.path.join(os.path.dirname(__file__), "document_en.md")
with open(markdown_file_en, "r") as file:
  markdown_text_en = file.read()
md_header_splits_en = mardown_splitter.split_text(markdown_text_en)

# Let's embed
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
my_activeloop_org_id = "learningprocess123"
my_activeloop_dataset_name = "my_dataset"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

db.add_documents(md_header_splits_ru)
db.add_documents(md_header_splits_en)
