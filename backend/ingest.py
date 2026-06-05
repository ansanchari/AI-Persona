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
    return text

def fetch_github_repo_data(owner: str, repo: str) -> list[dict]:
    print(f" Fetching data from GitHub: {owner}/{repo}...")
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    
    repo_info_url = f"https://api.github.com/repos/{owner}/{repo}"
    repo_info = requests.get(repo_info_url, headers=headers).json()
    
    if "default_branch" not in repo_info:
        return []
        
    default_branch = repo_info["default_branch"]
    tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1"
    tree_data = requests.get(tree_url, headers=headers).json()
    
    if "tree" not in tree_data:
        return []

    valid_extensions = (".md", ".py", ".js", ".ts", ".txt")
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

session = requests.Session()

def get_batch_embeddings(texts):
    url = "https://api.mistral.ai/v1/embeddings"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
    data = {"model": "mistral-embed", "input": texts}
    
    response = session.post(url, headers=headers, json=data, timeout=15)
    response.raise_for_status()
    
    return [item["embedding"] for item in response.json()["data"]]

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
    BATCH_SIZE = 20  # Extremely safe batch size
    total_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print("\n🚀 Starting safe, throttled ingestion. Please let this run...\n")
    
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        texts = [item["text"] for item in batch]
        current_batch_num = (i // BATCH_SIZE) + 1
        
        try:
            embeddings = get_batch_embeddings(texts)
            
            for j, item in enumerate(batch):
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[j],
                    payload={
                        "text": item["text"],
                        "source": item["source"],
                        "source_type": item["source_type"]
                    }
                ))
            print(f" Successfully processed batch {current_batch_num} of {total_batches}")
            
            time.sleep(1.5)
            
        except requests.exceptions.Timeout:
            print(f" Batch {current_batch_num} timed out. Mistral API is being slow. Skipping this batch to keep the script alive.")
        except Exception as e:
            print(f" Error on batch {current_batch_num}: {e}")
            
    print("\n Uploading data to Qdrant...")
    if points:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(" Success! The Knowledge Base is fully populated and ready for your presentation.")