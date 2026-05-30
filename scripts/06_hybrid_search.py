import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10   # беремо ширше, щоб RRF міг переранжувати

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

df["full_text"] = df["title"] + " " + df["abstract"]
tokenized_corpus = [doc.lower().split() for doc in df["full_text"].tolist()]
bm25 = BM25Okapi(tokenized_corpus)

def search_bm25(query, top_k = TOP_K):
    tokenized_query = query.lower().split(" ")
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices):
        results.append({
            "id": int(idx),
            "score": float(scores[idx]),
            "rank": rank + 1
        })
    return results

def search_vector(query, top_k = TOP_K):
    embedding = model.encode(query).tolist()
    response = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=False
    )
    
    results = []
    for rank, match in enumerate(response.matches):
        clean_id = match.id.replace("paper_", "")
        results.append({
            "id": int(clean_id),
            "score": float(match.score),
            "rank": rank + 1
        })
        
    return results                              


def search_hybrid(query, top_k = 5, rrf_k = 60):
    bm25_results = search_bm25(query, top_k = TOP_K)
    vector_results = search_vector(query, top_k = TOP_K)
    
    rrf_score = {}
    
    for item in bm25_results:
        doc_id = item["id"]
        rank = item["rank"]
        rrf_score[doc_id] = rrf_score.get(doc_id, 0) +  (1 / (rrf_k + rank))
        
    for item in vector_results:
        doc_id = item["id"]
        rank = item["rank"]
        rrf_score[doc_id] = rrf_score.get(doc_id, 0) +  (1 / (rrf_k + rank))

    sorted_docs = sorted(rrf_score.items(), key=lambda x: x[1], reverse=True)
    
    hybrid_results = []
    for rank, (doc_id, score) in enumerate(sorted_docs[:top_k]):
        hybrid_results.append({
            "id": int(doc_id),
            "rrf_score": float(score),
            "rank": rank + 1
        })
    return hybrid_results


queries = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text"
]
for q in queries:
    print(f"\n{'='*60}\nЗАПИТ: '{q}'\n{'='*60}")
    
    bm25_res_full = search_bm25(q, top_k=TOP_K)
    vector_res_full = search_vector(q, top_k=TOP_K)
    
    print("\n--- TOP-5 BM25 ---")
    for res in bm25_res_full[:5]:
        title = df.iloc[res["id"]]["title"]
        print(f"[{res['rank']}] Score: {res['score']:.4f} | {title[:70]}...")

    print("\n--- TOP-5 Vector ---")
    for res in vector_res_full[:5]:
        title = df.iloc[res["id"]]["title"]
        print(f"[{res['rank']}] Score: {res['score']:.4f} | {title[:70]}...")
        
    print("\n--- TOP-5 Hybrid (RRF) ---")
    for res in search_hybrid(q, top_k = 5):
        title = df.iloc[res["id"]]["title"]
        print(f"[{res['rank']}] Score: {res['rrf_score']:.4f} | {title[:70]}...")

