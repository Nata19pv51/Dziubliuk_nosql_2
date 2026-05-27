print("Починаємо імпорт бібліотек (це може зайняти хвилину)...")

from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os

print("Імпорт завершено успішно!")

# Load the file into a DataFrame
df = pd.read_parquet('data/arxiv_subset.parquet')
df_selsected_fields = df["title"] + " [SEP] " + df["abstract"]

model = SentenceTransformer('allenai/specter2_base')

texts_list = df_selsected_fields.tolist()

# 2. Embedding generation:
embeddings = model.encode(
    texts_list,
    batch_size=64,                # butch processing
    show_progress_bar=True,       # show progress
    normalize_embeddings=True     # L2-normalization
)

os.makedirs('embeddings', exist_ok=True)
np.save('embeddings/embeddings.npy', embeddings)

print(f"Загальна кількість оброблених текстів: {len(embeddings)}")
print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
print(f"Норма першого ембеддингу: {np.linalg.norm(embeddings[0]):.4f}")