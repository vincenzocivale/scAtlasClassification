import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import scipy.sparse as sp
import gc
from tqdm.auto import tqdm

# --- 1. CONFIGURAZIONE ---
# Lista di geni target a cui allineare il dataset
genes_list_path = "OS_scRNA_gene_index.19264.tsv"
# Colonna dei metadati da conservare
label_obs = 'cell_type'
# Soglia minima di geni espressi per conservare una cellula
n_genes_filter = 200
# Directory di input e output
root_dir = "/equilibrium/datasets/TCGA-histological-data/scDataset"
input_file = f"{root_dir}/nervous_system_sc_atlas.h5ad"
output_file = f"{root_dir}/nervous_system_sc_atlas_processed.h5ad"
# Dimensione dei chunk per l'elaborazione (da regolare in base alla RAM)
obs_chunk_size = 5000

# --- 2. CARICAMENTO DATI E METADATI ---
print("Caricamento della lista di geni target...")
genes_list = pd.read_csv(genes_list_path, header=None, sep="\t")[0].tolist()

print("Caricamento del dataset in modalità 'backed' (efficiente in memoria)...")
# 'backed=r' non carica l'intera matrice .X in memoria, ma solo su richiesta
adata_eye = sc.read_h5ad(input_file, backed='r')

# Assegna i nomi dei geni e li rende univoci
adata_eye.var_names = adata_eye.var['feature_name'].astype(str)
adata_eye.var_names_make_unique()
print(f"Dataset originale: {adata_eye.n_obs} cellule × {adata_eye.n_vars} geni")

# --- 3. PRE-FILTRAGGIO DELLE CELLULE (OTTIMIZZAZIONE CHIAVE) ---
# È molto più efficiente filtrare le cellule PRIMA del complesso processo di allineamento.
print(f"Identificazione delle cellule con almeno {n_genes_filter} geni espressi...")

n_genes_by_counts = np.zeros(adata_eye.n_obs, dtype=int)
# Calcoliamo i geni per cellula in chunk per non caricare tutta la matrice
for i in tqdm(range(0, adata_eye.n_obs, obs_chunk_size), desc="Calcolo geni per cellula"):
    chunk = adata_eye[i : i + obs_chunk_size].X
    # getnnz(axis=1) conta gli elementi non-zero per riga (cioè per cellula)
    n_genes_by_counts[i : i + chunk.shape[0]] = chunk.getnnz(axis=1)

# Crea una maschera booleana per le cellule da tenere
cells_to_keep_mask = n_genes_by_counts >= n_genes_filter
# Ottieni gli indici numerici delle cellule che passano il filtro
cell_indices_to_keep = np.where(cells_to_keep_mask)[0]

n_obs_before = adata_eye.n_obs
n_obs_after = len(cell_indices_to_keep)
print(f"Filtrate {n_obs_before - n_obs_after} cellule. Rimangono {n_obs_after} cellule valide.")

# --- 4. MAPPATURA DEGLI INDICI DEI GENI ---
print("Mappatura degli indici dei geni comuni tra dataset e lista target...")
common_genes = list(set(adata_eye.var_names) & set(genes_list))
print(f"Trovati {len(common_genes)} geni in comune su {len(genes_list)} richiesti.")

# Mappa: nome_gene -> indice nel dataset originale
gene_to_original_idx = {gene: i for i, gene in enumerate(adata_eye.var_names)}
original_indices = [gene_to_original_idx[gene] for gene in common_genes]

# Mappa: nome_gene -> indice nel nuovo dataset allineato
gene_to_aligned_idx = {gene: i for i, gene in enumerate(genes_list)}
aligned_indices = [gene_to_aligned_idx[gene] for gene in common_genes]
print(f"Verificati {len(original_indices)} geni pronti per la mappatura.")

# --- 5. PROCESSO DI ALLINEAMENTO IN CHUNK (SOLO SULLE CELLULE FILTRATE) ---
aligned_chunks = []
obs_list = []
n_vars_aligned = len(genes_list)

print("Avvio del processo di allineamento in chunk sulle cellule filtrate...")
# Iteriamo sugli indici delle cellule da tenere, non su tutto il dataset
for i in tqdm(range(0, n_obs_after, obs_chunk_size), desc="Allineamento chunk"):
    # Seleziona un chunk di indici di cellule valide
    chunk_indices = cell_indices_to_keep[i : i + obs_chunk_size]
    
    # Carica solo i dati necessari: le righe (cellule) e colonne (geni) richieste
    X_chunk_subset = adata_eye[chunk_indices, original_indices].X
    
    if sp.issparse(X_chunk_subset):
        X_chunk_subset = X_chunk_subset.toarray()
    
    # Crea la matrice di destinazione per questo chunk, inizializzata a zero
    X_chunk_aligned = np.zeros((len(chunk_indices), n_vars_aligned), dtype=X_chunk_subset.dtype)
    
    # Inserisce i valori dei geni comuni nelle posizioni corrette
    X_chunk_aligned[:, aligned_indices] = X_chunk_subset
    
    # Riconverte in formato sparse (efficiente) e aggiunge alla lista
    aligned_chunks.append(sp.csr_matrix(X_chunk_aligned))
    
    # Carica solo la colonna 'obs' richiesta per le cellule di questo chunk
    obs_list.append(adata_eye.obs.iloc[chunk_indices][[label_obs]])
    
    del X_chunk_subset, X_chunk_aligned
    gc.collect()

# --- 6. CREAZIONE E SALVATAGGIO DEL NUOVO OGGETTO Anndata ---
print("Concatenazione dei chunk processati...")
X_aligned = sp.vstack(aligned_chunks)
obs_aligned = pd.concat(obs_list, axis=0)

# Pulisce la memoria
del aligned_chunks, obs_list
gc.collect()

print("Creazione dell'oggetto AnnData finale allineato...")
adata_final = ad.AnnData(
    X=X_aligned,
    obs=obs_aligned,
    var=pd.DataFrame(index=genes_list)
)

print(f"Dataset finale creato: {adata_final.n_obs} cellule × {adata_final.n_vars} geni.")

print(f"Salvataggio del dataset processato in: {output_file}")
adata_final.write_h5ad(output_file)

# --- 7. VERIFICA FINALE ---
max_val = adata_final.X.max()
print(f"Valore massimo nella matrice di conte finali: {max_val}")
print("Processo completato con successo! 🎉")