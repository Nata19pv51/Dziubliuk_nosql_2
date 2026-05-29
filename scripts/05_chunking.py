import os
import re
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

def fixed_size_chunking(text, size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=lambda x: len(x.split()),
        separators=["\\n\\n", "\\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(text.strip())
    
    return chunks


def semantic_chunking(
        text: str,
        model: SentenceTransformer,
        threshold: float = 0.7,
        min_chunk_size: int = 50,
    ) -> List[str]:
    
    # Просте розділення на речення
    sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text.replace("\n", " ")) if s.strip()]
    
    if len(sentences) < 2:
        return sentences

    # Отримуємо ембеддинги речень
    embeddings = model.encode(sentences, normalize_embeddings=True)

    # Косинусна схожість між сусідніми реченнями
    similarities = [
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]
    # print(f"Подібності між реченнями: {[round(s, 3) for s in similarities]}")

    chunks, current_chunk = [], [sentences[0]]
    for i, sim in enumerate(similarities):
        if sim < threshold and len(" ".join(current_chunk)) >= min_chunk_size:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks


def process_and_upload(text, chunking_function, pinecone_index, desc=""):
    vector_to_upsert = []
    
    for _, row in tqdm(text.iterrows(), total=len(text), desc=desc):
        arxiv_id = str(row['id'])
        title = str(row['title'])
        abstract = str(row['abstract'])
        
        chunks = chunking_function(abstract)
    
        for i, chunk_text in enumerate(chunks):
            text_to_encode = f"{title} [SEP] {chunk_text}"
            chunk_embedding = model.encode(text_to_encode, normalize_embeddings=True).tolist()
            
            chunk_id = f"{arxiv_id}_chunk_{i}"
            chunk_metadata = {
                "arxiv_id": arxiv_id,
                "title": title,
                "chunk_text": chunk_text,
                "chunk_index": i,
                "year": int(row['year']),
                "category": str(row['category'])
            }

            vector_to_upsert.append({
                "id": chunk_id, 
                "values": chunk_embedding, 
                "metadata": chunk_metadata
            })
    BATCH_SIZE = 100    
    for i in range(0, len(vector_to_upsert), BATCH_SIZE):
        pinecone_index.upsert(vectors=vector_to_upsert[i : i + BATCH_SIZE])

def search_chunks(query, pinecone_index, index_name):
    print(f"\n--- Результати пошуку в індексі: {index_name} ---")
    query_vector = model.encode(f"{query} [SEP] ", normalize_embeddings=True).tolist()
    
    results = pinecone_index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True
    )
    
    for idx, match in enumerate(results["matches"], 1):
        meta = match["metadata"]
        score = match["score"]
        print(f"{idx}. [{score:.4f}] {meta['title']}")
        print(f"   Чанк {meta['chunk_index']}: {meta['chunk_text'][:150]}...\n")


df.loc[:, "length"] = df["abstract"].apply(lambda x: len(str(x).split()))
top_30_df = df.sort_values(by="length", ascending=False).head(30)
print(f"30 статей із найдовшими анотаціями: \n{top_30_df}")

text_abstract = top_30_df.iloc[0]["abstract"].strip()


INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"

existing_indexes = [idx["name"] for idx in pc.list_indexes()]

for idx_name in [INDEX_FIXED, INDEX_SEMANTIC]:
    if idx_name not in existing_indexes:
        print(f"Створюємо індекс {idx_name}...")
        pc.create_index(
            name=idx_name,
            dimension=VECTOR_DIM,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

idx_fixed = pc.Index(INDEX_FIXED)
idx_semantic = pc.Index(INDEX_SEMANTIC)

print("\nЗавантаження Fixed Chunks...")
process_and_upload(
    top_30_df, 
    lambda t: fixed_size_chunking(t, size=50, overlap=10), 
    idx_fixed, 
    desc="Fixed Chunking Progress"
)

print("\nЗавантаження Semantic Chunks...")
process_and_upload(
    top_30_df, 
    lambda t: semantic_chunking(t, model, threshold=0.6, min_chunk_size=100), 
    idx_semantic, 
    desc="Semantic Chunking Progress"
)


# Тестуємо пошук
test_queries = [
    "deep learning algorithms for object detection",
    "limitations of quantum computing"
]

for q in test_queries:
    print(f"\n{'='*60}\nЗАПИТ: '{q}'\n{'='*60}")
    search_chunks(q, idx_fixed, INDEX_FIXED)
    search_chunks(q, idx_semantic, INDEX_SEMANTIC)
