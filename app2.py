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

# ১. এনভায়রনমেন্ট সেটআপ
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

app = FastAPI()

# ২. স্ট্যাটিক ফাইল এবং টেমপ্লেট কনফিগারেশন
# নিশ্চিত করুন style_web.css ফাইলটি 'static' ফোল্ডারের ভেতর আছে
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# index_web.html ফাইলটি মেইন ডিরেক্টরিতে থাকলে directory="." হবে
templates = Jinja2Templates(directory=".")

# --- RAG লজিক (Website Data) ---

print("Loading website data... please wait.")
URL = "https://betopiagroup.com/"
loader = WebBaseLoader(URL)
docs = loader.load()

# টেক্সট স্প্লিটিং
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# ভেক্টর ডাটাবেস তৈরি
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

print("✅ Website Data Indexed and RAG Chain Ready!")

# --- Routes ---

# মূল পেজ লোড করার জন্য
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index_web.html", {"request": request})

# চ্যাটবক্স থেকে প্রশ্ন নেওয়ার জন্য (JSON Response দিবে)
@app.post("/ask")
async def ask_question(question: str = Form(...)):
    try:
        # AI থেকে উত্তর জেনারেট করা
        answer = rag_chain.invoke(question)
        # ফ্রন্টএন্ডের জাভাস্ক্রিপ্ট এই 'answer' কি-টি পড়বে
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"answer": "I'm sorry, I encountered an error while processing your request."}, status_code=500)

# --- রান করা ---
if __name__ == "__main__":
    import uvicorn
    # 8045 এর বদলে 8050 ব্যবহার করুন
    print("Starting server at http://127.0.0.1:8055")
    uvicorn.run(app, host="127.0.0.1", port=8055)