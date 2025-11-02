import argparse
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
import google.generativeai as genai
import os

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """Answer the question based on the context below:
{context}
---
Answer the question based on the context above:{question}"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text",type=str,nargs="?",default="Hello, what is RAG?",help="The query text")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    print("Loading embeddings...")
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Loading Chroma vector store...")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    print("Performing similarity search...")
    results = db.similarity_search_with_score(query_text, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    genai.configure(api_key = "AIzaSyBkS6BemzUZeJmUqoyPcW0YPaIoJtbJHOs")
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    print(response.text)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response}\nSources: {sources}"
    print(formatted_response)
    return formatted_response
    
if __name__ == "__main__":
    main()