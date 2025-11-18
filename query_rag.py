from langchain_chroma import Chroma
from langchain_classic.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from populate_db import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_rag(query_text:str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH,embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text,k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc,_score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context = context_text , question = query_text)

    model = OllamaLLM(model="deepseek-r1:1.5b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id") for doc,_score in results]
    formatted_response = f"Response : {response_text}\n Source: {sources}"
    return formatted_response

if __name__ == "__main__":
    quest = input(">>>")
    answer = query_rag(quest)
    print(answer)