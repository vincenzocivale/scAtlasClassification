import numpy as np
import anndata as ad
from datasets import Dataset, Features, Value, Sequence, ClassLabel, DatasetDict
import huggingface_hub
import os

# --- 1. CONFIGURAZIONE ---
EMBEDDINGS_PATH = "/data2/home/vcivale/scTumorClassification/embeddings/lung_tumor_01B-resolution_singlecell_cell_embedding_t4_resolution.npy"
ADATA_PATH = "dataset/lung_cell_atlas_processed.h5ad"

DISEASE_COLUMN = 'disease'
CELL_TYPE_COLUMN = 'cell_type'

HF_USERNAME = "Yuto2007"
DATASET_NAME = "lung-tumor-embeddings"

# --- 2. CARICAMENTO DEI DATI ---
print("Caricamento dati in corso...")
embeddings = np.load(EMBEDDINGS_PATH)
adata = ad.read_h5ad(ADATA_PATH)
print("✅ Dati caricati con successo.")

# --- 3. CONTROLLO DI COERENZA ---
if embeddings.shape[0] != adata.n_obs:
    print(f"❌ Errore critico: Il numero di embedding ({embeddings.shape[0]}) non corrisponde al numero di cellule ({adata.n_obs}).")
    exit()
else:
    print("✅ Coerenza verificata.")

# --- 💡 FILTRAGGIO: rimuovi 'unknown' e classi troppo piccole ---
print("\nFiltraggio delle cellule con 'cell_type' == 'unknown' e delle classi con meno di 2 campioni...")

original_cell_count = adata.n_obs

# 1️⃣ Rimuovi le cellule con cell_type == 'unknown'
unknown_mask = adata.obs[CELL_TYPE_COLUMN] == 'unknown'
num_unknown = np.sum(unknown_mask)

if num_unknown > 0:
    print(f"Rimozione di {num_unknown} cellule con cell_type == 'unknown'")
    keep_mask = ~unknown_mask
    adata = adata[keep_mask].copy()
    embeddings = embeddings[keep_mask]
else:
    print("Nessuna cellula con cell_type == 'unknown' trovata.")

# 2️⃣ Rimuovi le classi con meno di 2 campioni
cell_type_counts = adata.obs[CELL_TYPE_COLUMN].value_counts()
types_to_keep = cell_type_counts[cell_type_counts >= 2].index.tolist()

keep_mask = adata.obs[CELL_TYPE_COLUMN].isin(types_to_keep)
adata = adata[keep_mask].copy()
embeddings = embeddings[keep_mask]

print(f"Cellule rimosse in totale: {original_cell_count - adata.n_obs}")
print(f"Numero di cellule rimanenti: {adata.n_obs}")

# --- 4. PREPARAZIONE DEL DATASET ---
print("\nPreparazione del dataset per Hugging Face...")
dataset_data = {
    'cell_id': adata.obs_names.to_list(),
    'embedding': embeddings.tolist(),
    'disease': adata.obs[DISEASE_COLUMN].to_list(),
    'cell_type': adata.obs[CELL_TYPE_COLUMN].to_list()
}

# --- 5. CASTING DELLE LABEL E DEFINIZIONE DELLO SCHEMA ---
disease_labels = sorted(adata.obs[DISEASE_COLUMN].unique().tolist())
cell_type_labels = sorted(adata.obs[CELL_TYPE_COLUMN].unique().tolist())

features = Features({
    'cell_id': Value('string'),
    'embedding': Sequence(Value('float32')),
    'disease': ClassLabel(names=disease_labels),
    'cell_type': ClassLabel(names=cell_type_labels)
})

hf_dataset = Dataset.from_dict(dataset_data, features=features)
print("✅ Label convertite e dataset strutturato.")

# --- 6. SPLIT STRATIFICATO DEL DATASET ---
print("\nInizio split stratificato del dataset...")
train_test_split_dict = hf_dataset.train_test_split(test_size=0.30, stratify_by_column='cell_type', seed=42)
train_dataset = train_test_split_dict['train']
test_validation_dataset = train_test_split_dict['test']

validation_test_split_dict = test_validation_dataset.train_test_split(test_size=0.50, stratify_by_column='cell_type', seed=42)
validation_dataset = validation_test_split_dict['train']
test_dataset = validation_test_split_dict['test']

final_dataset = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})
print("✅ Split completato con successo.")

# --- 7. PUBBLICAZIONE SU HUGGING FACE HUB ---
repo_id = f"{HF_USERNAME}/{DATASET_NAME}"
print(f"\nInizio pubblicazione del dataset splittato su Hugging Face Hub: {repo_id}")
final_dataset.push_to_hub(repo_id=repo_id, private=False)
print(f"\n🎉 Pubblicazione completata con successo! Link: https://huggingface.co/datasets/{repo_id}")
