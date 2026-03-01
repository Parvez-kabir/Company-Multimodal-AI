import os
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ১. এনভায়রনমেন্ট ও সেটিংস লোড করা
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ওয়েবসাইট ব্লকিং এড়াতে User Agent সেট করা হয়েছে
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

app = FastAPI()

# ২. স্ট্যাটিক ফাইল ও টেমপ্লেট সেটআপ
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=".")

# --- ৩. RAG লজিক (Website & PDF Indexing) ---

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = OpenAIEmbeddings()

print("Loading Betopia website data... please wait.")
URL = "https://betopiagroup.com/"
loader = WebBaseLoader(URL)
docs = loader.load()
splits = text_splitter.split_documents(docs)

# প্রাথমিক ভেক্টর ডাটাবেস তৈরি
vector_db = FAISS.from_documents(splits, embeddings)

# LLM এবং প্রম্পট সেটআপ
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# চেইন জেনারেট করার ফাংশন
def get_rag_chain():
    retriever = vector_db.as_retriever()
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )

rag_chain = get_rag_chain()

# --- ৪. Routes (এপিআই এন্ডপয়েন্ট) ---

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("indexmultimodel.html", {"request": request})

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    try:
        # AI থেকে উত্তর জেনারেট করা
        answer = rag_chain.invoke(question)
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        return JSONResponse(content={"answer": f"Processing Error: {str(e)}"}, status_code=500)

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # ফাইলটি সাময়িকভাবে লোকাললি সেভ করা
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # পিডিএফ থেকে ডেটা লোড এবং স্প্লিট করা
        pdf_loader = PyPDFLoader(temp_path)
        pdf_docs = pdf_loader.load()
        pdf_splits = text_splitter.split_documents(pdf_docs)
        
        # ভেক্টর ডাটাবেসে নতুন ডকুমেন্ট যুক্ত করা
        vector_db.add_documents(pdf_splits)
        
        # গ্লোবাল চেইন আপডেট করা যাতে নতুন পিডিএফ থেকে উত্তর দিতে পারে
        global rag_chain
        rag_chain = get_rag_chain()
        
        # টেম্পোরারি ফাইল মুছে ফেলা
        os.remove(temp_path)
        
        # এরর ফিক্স: file.name এর পরিবর্তে file.filename ব্যবহার করা হয়েছে
        return JSONResponse(content={"answer": f"Successfully indexed '{file.filename}'. Now I can answer questions from this document as well."})
    
    except Exception as e:
        return JSONResponse(content={"answer": f"Upload failed: {str(e)}"}, status_code=500)

# --- ৫. সার্ভার রান করা ---
if __name__ == "__main__":
    import uvicorn
    print("Starting Ultra Pro AI Server at http://127.0.0.1:8075")
    uvicorn.run(app, host="127.0.0.1", port=8075)