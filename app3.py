import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ১. এনভায়রনমেন্ট ও সেটিংস
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

app = FastAPI()

# ২. স্ট্যাটিক ফাইল ও টেমপ্লেট সেটআপ
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# indexvoice.html যদি মেইন ফোল্ডারে থাকে তবে directory="."
templates = Jinja2Templates(directory=".")

# --- ৩. RAG লজিক (Website Data Indexing) ---

print("Loading website data... please wait.")
URL = "https://betopiagroup.com/"
loader = WebBaseLoader(URL)
docs = loader.load()

# টেক্সট স্প্লিটিং
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# ভেক্টর ডাটাবেস তৈরি (FAISS)
embeddings = OpenAIEmbeddings()
vector_db = FAISS.from_documents(splits, embeddings)
retriever = vector_db.as_retriever()

# LLM এবং প্রম্পট সেটআপ
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# চেইন তৈরি
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt 
    | llm 
    | StrOutputParser()
)

print("✅ Website Data Indexed and Voice-Ready Chain Active!")

# --- ৪. Routes (এপিআই এন্ডপয়েন্ট) ---

# মেইন ইন্টারফেস লোড করার জন্য
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    # এখানে indexvoice.html কল করা হয়েছে
    return templates.TemplateResponse("indexvoice.html", {"request": request})

# ভয়েস বা টেক্সট প্রশ্নের উত্তর দেওয়ার জন্য
@app.post("/ask")
async def ask_question(question: str = Form(...)):
    try:
        # AI থেকে উত্তর জেনারেট করা
        answer = rag_chain.invoke(question)
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"answer": "I'm sorry, I couldn't process that."}, status_code=500)

# --- ৫. সার্ভার রান করা ---
if __name__ == "__main__":
    import uvicorn
    # নতুন পোর্ট ৮০৫৫ ব্যবহার করা হয়েছে
    print("Starting server at http://127.0.0.1:8060")
    uvicorn.run(app, host="127.0.0.1", port=8060)