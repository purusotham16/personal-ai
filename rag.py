import chromadb
from dotenv import load_dotenv
import os

# ==========================================
# LOAD ENVIRONMENT
# ==========================================

load_dotenv()

# ==========================================
# CREATE PERSISTENT CHROMADB
# ==========================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

# ==========================================
# CREATE / GET COLLECTION
# ==========================================

collection = chroma_client.get_or_create_collection(
    name="personal_documents"
)

# ==========================================
# LOAD DOCUMENT
# ==========================================

document_path = "documents/about_me.txt"

if os.path.exists(document_path):

    with open(
        document_path,
        "r",
        encoding="utf-8"
    ) as file:

        document_text = file.read()

else:

    document_text = ""

# ==========================================
# SPLIT DOCUMENT INTO CHUNKS
# ==========================================

chunks = []

chunk_size = 500

for i in range(0, len(document_text), chunk_size):

    chunk = document_text[i:i + chunk_size]

    chunks.append(chunk)

# ==========================================
# STORE DOCUMENT CHUNKS
# ==========================================

if chunks:

    collection.upsert(
        documents=chunks,
        ids=[
            f"chunk_{i}"
            for i in range(len(chunks))
        ]
    )

# ==========================================
# RETRIEVE RELEVANT INFORMATION
# ==========================================

def retrieve(query):

    if not query:
        return ""

    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    if not results["documents"]:
        return ""

    relevant_documents = results["documents"][0]

    context = "\n\n".join(
        relevant_documents
    )

    return context