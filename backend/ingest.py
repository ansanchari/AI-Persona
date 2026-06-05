import os
import uuid
import time
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")

TARGET_REPOS = [
    "ansanchari/ProAcquis",
    "ansanchari/Aura",
    "ansanchari/Portfolio",
    "ansanchari/AI-Video-Automation",
    "ansanchari/Empathia",
    "ansanchari/ansanchari",
    "ansanchari/Hotel-Bot",
    "ansanchari/Imperial-Decryptor",
    "ansanchari/TTT",
    "ansanchari/SWatch",
    "ansanchari/ResponsiveLoginPage",
    "ansanchari/H_37_Sanchari-Mukherjee"
]

try:
    qdrant_client = QdrantClient(path="local_qdrant_db")
except Exception as e:
    print(f"Could not connect to Qdrant: {e}")
    exit(1)

def create_collection_if_not_exists(collection_name: str):
    collections = qdrant_client.get_collections().collections
    if not any(c.name == collection_name for c in collections):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        print(f"Collection '{collection_name}' created.")

def extract_resume_text(pdf_path: str) -> str:
    print(f" Extracting text from {pdf_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
        
    system_override = """
    SYSTEM NOTE: When discussing projects, the architectural narrative strictly focuses on Scalability & Optimization. 
    Additionally, note that the publishing/deployment portion of the projects were not completed in reality and remain local implementations.
    """
    return text + "\n" + system_override

def fetch_github_repo_data(owner: str, repo: str) -> list[dict]:
    print(f" Fetching data from GitHub: {owner}/{repo}...")
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    
    repo_info_url = f"https://api.github.com/repos/{owner}/{repo}"
    repo_info = requests.get(repo_info_url, headers=headers).json()
    
    if "default_branch" not in repo_info:
        print(f"  -> Skipping {repo}: Could not fetch repository details.")
        return []
        
    default_branch = repo_info["default_branch"]
    
    tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1"
    tree_data = requests.get(tree_url, headers=headers).json()
    
    if "tree" not in tree_data:
        return []

    valid_extensions = (".md", ".py", ".js", ".ts", ".txt", ".json")
    files_data = []
    
    for item in tree_data["tree"]:
        if item["type"] == "blob" and item["path"].endswith(valid_extensions):
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}/{item['path']}"
            raw_resp = requests.get(raw_url, headers=headers)
            if raw_resp.status_code == 200:
                files_data.append({
                    "path": item["path"],
                    "content": raw_resp.text,
                    "source_type": "github",
                    "repo_name": repo
                })
    return files_data

def get_mistral_embedding(text: str) -> list[float]:
    url = "https://api.mistral.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-embed",
        "input": [text]
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

if __name__ == "__main__":
    COLLECTION_NAME = "my_persona_data"
    create_collection_if_not_exists(COLLECTION_NAME)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_chunks = []
    
    if os.path.exists("resume.pdf"):
        resume_text = extract_resume_text("resume.pdf")
        chunks = splitter.split_text(resume_text)
        for chunk in chunks:
            all_chunks.append({"text": chunk, "source": "resume.pdf", "source_type": "resume"})
    else:
        print(" Warning: resume.pdf not found in directory.")

    for repo_path in TARGET_REPOS:
        owner, repo_name = repo_path.split("/") 
        
        repo_files = fetch_github_repo_data(owner, repo_name)
        
        for file in repo_files:
            chunks = splitter.split_text(file["content"])
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk, 
                    "source": file["path"], 
                    "source_type": "github", 
                    "repo_name": repo_name
                })
                
    print(f"\n Total chunks to embed: {len(all_chunks)}")
    
    points = []
    for i, item in enumerate(all_chunks):
        print(f"Embedding chunk {i+1}/{len(all_chunks)}...", end="\r")
        vector = get_mistral_embedding(item["text"])
        
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": item["text"],
                "source": item["source"],
                "source_type": item["source_type"]
            }
        ))
        time.sleep(0.2) 
        
    print("\n Uploading to Qdrant Database...")
    if points:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(" Success! The Knowledge Base is fully populated.")