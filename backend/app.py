

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import re
from collections import defaultdict

# Globals
faiss_db = None
qa_chain = None
all_docs = []
year_event_map = {}  # new: year -> list of events


def load_and_split_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(docs)


def create_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)
    
def create_qa_chain(faiss_index):
    retriever = faiss_index.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY  
    )
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")


def answer_question(question):
    global qa_chain
    if qa_chain is None:
        return "❌ Please upload and process a PDF first."
    try:
        result = qa_chain.run(question)
        return result
    except Exception as e:
        return f"❌ Error answering question: {str(e)}"
    
    def show_timeline(): 
     global all_docs

    if not all_docs:
        return "❌ Please upload and process a PDF first."

    from collections import defaultdict
    import re

    year_event_map = defaultdict(list)
    pattern = r"\b(1[5-9]\d{2}|20\d{2})\b"  # Match years from 1500–2099

    for doc in all_docs:
        sentences = re.split(r'(?<=[.!?])\s+', doc.page_content)
        for sentence in sentences:
            sentence = sentence.strip()

            # Skip short or noisy sentences
            if len(sentence) < 30 or re.search(r'\b(page|figure|table|chapter)\b', sentence, re.IGNORECASE):
                continue

            matches = re.findall(pattern, sentence)
            for year in matches:
                year_event_map[int(year)].append(sentence)

    if not year_event_map:
        return "⚠️ No historical events with years were found in the document."

    # Format timeline with better structure
    timeline = []
    timeline.append("📜 Chronological Timeline of Events")
    timeline.append("=" * 40)

    for year in sorted(year_event_map.keys()):
        timeline.append(f"\n🗓 Year: {year}")
        timeline.append("-" * 40)
        events = list(dict.fromkeys(year_event_map[year]))[:2]  # Remove duplicates, max 2
        for i, event in enumerate(events, start=1):
            timeline.append(f"{i}) {event}")
    
    return "\n".join(timeline)
import os

def upload_pdf(file):
    global faiss_db, qa_chain, all_docs

    try:
        if file is None:
            return "❌ No file uploaded."

        file_path = file.name
        if not os.path.exists(file_path):
            return f"❌ File not found: {file_path}"

        all_docs = load_and_split_pdf(file_path)
        faiss_db = create_faiss_index(all_docs)
        qa_chain = create_qa_chain(faiss_db)

        return f"✅ Processed file: {os.path.basename(file_path)}"

    except Exception as e:
        return f"❌ Error: {str(e)}"
    
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import re
from collections import defaultdict
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI()


# Serve static files from /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at root
from fastapi.responses import FileResponse
@app.get("/")
def read_root():
    return FileResponse("static/index.html")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
faiss_db = None
qa_chain = None
all_docs = []

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global faiss_db, qa_chain, all_docs

    try:
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Load PDF and extract pages
        loader = PyMuPDFLoader(tmp_path)
        all_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(all_docs)

        # ✅ IMPROVED HISTORICAL CONTENT CHECK
        sample_text = " ".join([doc.page_content for doc in all_docs[:5]]).lower()

        historical_keywords = [
            "independence", "revolution", "empire", "freedom", "war", "colony", "partition",
            "battle", "constitution", "civilization", "ancient", "medieval", "dynasty",
            "treaty", "british raj", "mughal", "gupta", "maurya", "nationalist", "reform",
            "movement", "invasion", "struggle", "timeline", "historical", "history of"
        ]

        match_count = sum(1 for kw in historical_keywords if kw in sample_text)

        if match_count < 3:
            return JSONResponse(
                status_code=400,
                content={"error": "❌ This PDF does not appear to contain strong historical content."}
            )

        # Proceed with processing
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        faiss_db = FAISS.from_documents(chunks, embeddings)

        retriever = faiss_db.as_retriever(search_kwargs={"k": 5})
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.2,
            google_api_key=GOOGLE_API_KEY
        )
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

        return {"message": "✅ PDF processed successfully."}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



@app.post("/ask")
async def ask_question(question: str = Form(...)):
    global qa_chain
    if qa_chain is None:
        return JSONResponse(status_code=400, content={"error": "❌ Please upload and process a PDF first."})
    try:
        answer = qa_chain.run(question)
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/timeline")
async def get_timeline():
    global all_docs
    if not all_docs:
        return JSONResponse(status_code=400, content={"error": "❌ Please upload and process a PDF first."})

    year_event_map = defaultdict(list)
    pattern = r"\b(1[5-9]\d{2}|20\d{2})\b"

    for doc in all_docs:
        sentences = re.split(r'(?<=[.!?])\s+', doc.page_content)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 30 or re.search(r'\b(page|figure|table|chapter)\b', sentence, re.IGNORECASE):
                continue
            matches = re.findall(pattern, sentence)
            for year in matches:
                year_event_map[int(year)].append(sentence)

    if not year_event_map:
        return {"timeline": "⚠️ No historical events with years were found in the document."}

    timeline = []
    for year in sorted(year_event_map.keys()):
        events = list(dict.fromkeys(year_event_map[year]))[:2]
        timeline.append({"year": year, "events": events})

    return {"timeline": timeline}



