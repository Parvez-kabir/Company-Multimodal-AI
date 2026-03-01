import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain components
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Environment variables load kora
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("ERROR: OPENAI_API_KEY khuje paua jayni! .env file check koren.")

app = FastAPI()

# 2. Static folder check (static folder na thakle error jeno na dey)
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 3. RAG Pipeline Initialization
# Note: PDF file-er path-ti apnar notebook onujayi deya hoyeche
file_path = "data/Document1.pdf"

if os.path.exists(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(docs, embeddings)
    retriever = vector_db.as_retriever()
    print("PDF Loaded and Vector DB Ready!")
else:
    print(f"ERROR: {file_path} eai path-e kono PDF paua jayni!")

# 4. LLM & Chain Setup
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Data Model
class ChatQuery(BaseModel):
    message: str

# 6. Routes
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>index.html file-ti folder-e khuje paua jayni!</h1>"

@app.post("/chat")
async def chat(query: ChatQuery):
    try:
        response = rag_chain.invoke(query.message)
        return {"reply": response}
    except Exception as e:
        return {"reply": f"Error hoyeche: {str(e)}"}

# 7. SERVER RUN (Eita thakle 'python app.py' dile server start hobe)
if __name__ == "__main__":
    import uvicorn
    print("--- Server starting at http://127.0.0.1:8000 ---")
    uvicorn.run(app, host="127.0.0.1", port=8000)