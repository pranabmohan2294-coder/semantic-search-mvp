from sentence_transformers import SentenceTransformer
import chromadb
import os
import re

# -----------------------------
# Load embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Initialize Chroma vector DB
# -----------------------------
client = chromadb.Client()

collection = client.create_collection(
    name="policy_documents"
)

# -----------------------------
# Function: semantic chunking
# -----------------------------
def chunk_document(text):

    # Split by section titles
    sections = re.split(r'Section \d+:', text)

    chunks = []

    for section in sections:
        cleaned = section.strip()

        if len(cleaned) > 100:   # ignore tiny chunks
            chunks.append(cleaned)

    return chunks


# -----------------------------
# Load documents
# -----------------------------
data_folder = "data"

documents = []
metadatas = []
ids = []

doc_counter = 0

for filename in os.listdir(data_folder):

    if not filename.endswith(".txt"):
        continue

    file_path = os.path.join(data_folder, filename)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chunks = chunk_document(text)

    for chunk_index, chunk in enumerate(chunks):

        documents.append(chunk)

        # metadata helps retrieval later
        metadatas.append({
            "source_document": filename,
            "chunk_number": chunk_index
        })

        ids.append(f"{filename}_{chunk_index}")

        doc_counter += 1


print(f"Loaded {doc_counter} chunks into vector database")

# -----------------------------
# Create embeddings
# -----------------------------
embeddings = model.encode(documents).tolist()

# -----------------------------
# Store in Chroma
# -----------------------------
collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print("Documents indexed successfully.")

# -----------------------------
# Query loop
# -----------------------------
while True:

    query = input("\nEnter your search query (or type 'exit'): ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    print("\nTop Results:\n")

    for i, doc in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][i]

        print(f"Result {i+1}")
        print("Source:", metadata["source_document"])
        print(doc[:500])
        print("-----------")