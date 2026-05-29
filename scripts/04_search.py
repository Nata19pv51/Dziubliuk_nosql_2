import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm


load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

# 1. Підключення до індексу arxiv-papers у Pinecone і завантажити модель allenai/specter2_base
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
# Читаємо локальний файл з текстами
df = pd.read_parquet("data/arxiv_subset.parquet")

# 2. Функція кодування запиту в ембеддинг
def encode_query(query: str):
    # Додаємо " [SEP] " - так вимагає specter2_base, щоб відрізнити запит від документа
    formatted_query = query + " [SEP] "
    query_embedding = model.encode(formatted_query, normalize_embeddings=True)
    return query_embedding.tolist()

def print_results(results, title="Результати пошуку:"):
    print(f"\n{'='*50}\n{title}\n{'='*50}")
    for match in results["matches"]:
        metadata = match["metadata"]
        score = match["score"]
        full_abstract = df[df["id"] == metadata["arxiv_id"]]["abstract"].values[0]
        
        print(f"[{score:.4f}] {metadata['title']}")
        print(f"Категорія: {metadata['category']} | Рік: {metadata['year']}")
        print(f"Абстракт: {full_abstract[:150]}...\n")
        
        
query_text = "teaching machines to recognize objects in pictures"
query_vector = encode_query(query_text)

# 3. Чистий семантичний пошук
# Робимо запит до Pinecone, щоб знайти схожі статті
results_basic = index.query(
    vector=query_vector, 
    top_k=TOP_K,
    include_metadata=True 
)
# print(results_basic)

print("Найновіша стаття в датасеті:", df['year'].max())

# 4. Пошук з фільтрацією
# Шукаємо схожі статті на тему Reinforcement Learning...
query_rl = "reinforcement learning in robotics"
vector_rl = encode_query(query_rl)

results_filtered_A = index.query(
    vector=vector_rl,
    top_k=TOP_K,
    include_metadata=True,
    filter={
        "year": {"$gte": 2019},
        "category": {"$eq": "cs.LG"}
    }
)
print_results(results_filtered_A, "Фільтр А (>= 2019, cs.LG)")
print("Перевірка Фільтру А за допомогою pandas")
matches_filter_A = len(df[(df['year'] >= 2019) & (df['category'] == 'cs.LG')])
print(f"Локально знайдено статей для Фільтру А: {matches_filter_A}")

results_filtered_B = index.query(
    vector=vector_rl,
    top_k=TOP_K,
    include_metadata=True,
    filter={
        "year": {"$lt": 2015}
    }
)
print_results(results_filtered_B, "Фільтр B (< 2015)")
print("Перевірка Фільтру B за допомогою pandas")
matches_filter_B = len(df[(df['year'] < 2015)])
print(f"Локально знайдено статей для Фільтру B: {matches_filter_B}")


# 5. Порівняти різні метрики схожості на локальних ембеддингах
print("****************** Метрики схожості на локальних ембеддингах *******************")

# Завантажуємо всі ембеддинги
all_embeddings = np.load("embeddings/embeddings.npy")
query_vec_np = np.array(query_vector) # вектор з першого пошуку

# Обчислення Dot Product (скалярний добуток)
dot_products = np.dot(all_embeddings, query_vec_np)

# Обчислення Cosine Similarity (косинусна подібність)
norms_db = norm(all_embeddings, axis=1)
norm_query = norm(query_vec_np)
print(norm_query)
cosine_similarities = dot_products / (norms_db * norm_query)

# Обчислення L2-distance (евклідова відстань)
l2_distances = norm(all_embeddings - query_vec_np, axis=1)

def print_local_top(scores, reverse=False, title=""):
    print(f"\n--- {title} ---")
    # Якщо шукаємо відстань, нам потрібні найменші значення (від початку)
    # Якщо схожість - найбільші (з кінця, тому беремо -5:)
    if reverse:
        top_indices = np.argsort(scores)[:TOP_K]
    else:
        top_indices = np.argsort(scores)[-TOP_K:][::-1]
        
    for i in top_indices:
        score = scores[i]
        title = df.iloc[i]["title"]
        print(f"[{score:.4f}] {title[:80]}...")

print_local_top(cosine_similarities, title="Топ-5: Cosine Similarity")
print_local_top(dot_products, title="Топ-5: Dot Product")
print_local_top(l2_distances, reverse=True, title="Топ-5: L2 Distance (the smallest values)")