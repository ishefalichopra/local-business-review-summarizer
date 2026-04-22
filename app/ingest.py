import json
import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

load_dotenv()

# Config 
QDRANT_HOST        = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT        = int(os.getenv("QDRANT_PORT", 6333))
BUSINESS_FILE      = os.getenv("BUSINESS_FILE", "data/yelp_academic_dataset_business.json")
REVIEW_FILE        = os.getenv("REVIEW_FILE",  "data/yelp_academic_dataset_review.json")
MAX_BUSINESSES     = int(os.getenv("MAX_BUSINESSES", 50))
MAX_REVIEWS        = int(os.getenv("MAX_REVIEWS_PER_BUSINESS", 20))
COLLECTION_NAME    = "business_reviews"
VECTOR_SIZE        = 384

# Clients 
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to Qdrant...")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Text splitter 
splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=32
)

# Step 1: Load businesses 
def load_businesses(max_count):
    businesses = {}
    print(f"Loading up to {max_count} restaurant businesses...")
    with open(BUSINESS_FILE, "r") as f:
        for line in f:
            if len(businesses) >= max_count:
                break
            biz = json.loads(line.strip())
            if (
                biz.get("is_open") == 1
                and biz.get("review_count", 0) >= 10
                and biz.get("categories")
            ):
                businesses[biz["business_id"]] = {
                    "business_id": biz["business_id"],
                    "name":        biz["name"],
                    "city":        biz["city"],
                    "state":       biz["state"],
                    "stars":       biz["stars"],
                    "categories":  biz["categories"],
                }
    print(f"Loaded {len(businesses)} businesses")
    return businesses

# Step 2: Load reviews for those businesses 
def load_reviews(business_ids, max_per_business):
    reviews = {}
    for bid in business_ids:
        reviews[bid] = []

    print(f"Scanning reviews file (this takes ~30 seconds on 5GB)...")
    with open(REVIEW_FILE, "r") as f:
        for line in tqdm(f, desc="Reading reviews"):
            rev = json.loads(line.strip())
            bid = rev["business_id"]
            if bid in reviews and len(reviews[bid]) < max_per_business:
                reviews[bid].append({
                    "review_id": rev["review_id"],
                    "stars":     rev["stars"],
                    "date":      rev["date"],
                    "text":      rev["text"].strip(),
                })
    return reviews

# Step 3: Create Qdrant collection 
def setup_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists, deleting and recreating...")
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' created")

# Step 4: Embed and store 
def embed_and_store(businesses, reviews):
    points = []
    for bid, biz in tqdm(businesses.items(), desc="Embedding reviews"):
        biz_reviews = reviews.get(bid, [])
        if not biz_reviews:
            continue
        for rev in biz_reviews:
            if len(rev["text"].split()) < 5:
                continue
            chunks = splitter.split_text(rev["text"])
            for chunk in chunks:
                vector = embedder.encode(chunk).tolist()
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "business_id":   bid,
                        "business_name": biz["name"],
                        "city":          biz["city"],
                        "state":         biz["state"],
                        "biz_stars":     biz["stars"],
                        "categories":    biz["categories"],
                        "review_id":     rev["review_id"],
                        "review_stars":  rev["stars"],
                        "review_date":   rev["date"],
                        "text":          chunk,
                    }
                ))
            if len(points) >= 100:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                points = []
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Done! Total chunks stored in Qdrant.")

# Main
if __name__ == "__main__":
    businesses = load_businesses(MAX_BUSINESSES)
    reviews    = load_reviews(set(businesses.keys()), MAX_REVIEWS)
    setup_collection()
    embed_and_store(businesses, reviews)
    info = client.get_collection(COLLECTION_NAME)
    print(f"\n Ingestion complete!")
    print(f"   Businesses : {len(businesses)}")
    print(f"   Vectors    : {info.points_count}")
