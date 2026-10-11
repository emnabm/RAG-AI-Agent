from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os
import time


load_dotenv()
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


EMBED_MODEL="gemini-embedding-001"
EMBED_DIM=3072
splitter= SentenceSplitter(chunk_size=1000,chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    pdf_reader= PDFReader()
    docs= pdf_reader.load_data(file=path)
    texts=[d.text for d in docs if getattr(d,"text",None)]
    chunks= []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i + 50]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
        time.sleep(1)  # avoid 429 errors
    return vectors


# def embed_texts(texts: list[str]) -> list[list[float]]:
#     response= client.embeddings.create(
#         model=EMBED_MODEL,
#         input=texts,
#     )
#     return [item.embedding for item in response.data]
  