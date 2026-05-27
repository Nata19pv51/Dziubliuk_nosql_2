import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200   # Pinecone рекомендує батчі до 200 векторів

# Ініціалізація клієнта
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Список існуючих індексів
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

# Створюємо новий індекс, якщо він не існує
if INDEX_NAME not in existing_indexes:
    print(f"Індекс '{INDEX_NAME}' не знайдено. Створюємо новий...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="dotproduct",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Індекс успішно створено!")
else:
    print(f"Індекс '{INDEX_NAME}' вже існує.")

# Підключаємось до індексу для подальшого завантаження даних
index = pc.Index(INDEX_NAME)

# Завантаження датасету з файлу (метадані)
df = pd.read_parquet(INPUT_PARQUET)

# Завантаження ембеддингів з файлу
embeddings = np.load(INPUT_EMBEDDINGS)

total_records = len(df)

for i in range(0, total_records, BATCH_SIZE):
    df_batch = df.iloc[i : i + BATCH_SIZE]
    emb_batch = embeddings[i : i + BATCH_SIZE]
    
    vectors_to_upsert = []
    
    # Проходимо по кожному елементу батчу
    for j in range(len(df_batch)):
        row = df_batch.iloc[j]
        vector = emb_batch[j].tolist()

        metadata = {
            "arxiv_id": str(row["id"]),
            "title": str(row["title"]),
            "abstract": str(row["abstract"])[:500],
            "authors": str(row["authors"])[:200],
            "year": int(row["year"]),
            "category": str(row["category"])
        }
        
        # Обчислюємо глобальний номер статті для створення унікального ID (напр. "paper_0", "paper_205")
        article_number = i + j
        
        # Додаємо сформований об'єкт у наш список для відправки
        vectors_to_upsert.append({
            "id": f"paper_{article_number}",
            "values": vector,
            "metadata": metadata
        })
    
    # Відправляємо готову пачку з 200 векторів у Pinecone
    index.upsert(vectors=vectors_to_upsert)
    stats = index.describe_index_stats()
    print(f"Загальна кількість векторів в індексі: {stats.total_vector_count}")