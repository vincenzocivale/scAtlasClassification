import scanpy as sc
import pandas as pd
from scipy.sparse import csc_matrix
import numpy as np
import anndata as ad
from tqdm import tqdm
from scipy.sparse import lil_matrix, hstack
import scipy.sparse as sp
from tqdm import tqdm

# read tsv file with gene names
genes_list = pd.read_csv("OS_scRNA_gene_index.19264.tsv", header=None, sep="\t")[0].tolist()

label_obs = 'cell_type'
n_genes_filter = 200

root_dir = "/equilibrium/datasets/TCGA-histological-data/scDataset"

adata_eye = sc.read_h5ad(f"{root_dir}/eye_sc_atlas.h5ad", backed='r')

# Assign the gene names from the 'feature_name' column
adata_eye.var_names = adata_eye.var['feature_name'] # type: ignore

# Convert the index from Categorical to string (object) type
adata_eye.var.index = adata_eye.var.index.astype(str)

adata_eye.var_names_make_unique()

print("Starting alignment of eye dataset to provided gene list...")

n_obs = adata_eye.n_obs
n_vars_aligned = len(genes_list)
X_aligned = sp.csr_matrix((n_obs, n_vars_aligned), dtype=adata_eye.X.dtype)

adata_eye_aligned = ad.AnnData(
    X=X_aligned,
    obs=adata_eye.obs,
    var=pd.DataFrame(index=genes_list)
)

common_genes = list(set(adata_eye.var_names) & set(genes_list))

print("Starting alignment of eye dataset to provided gene list...")

n_obs = adata_eye.n_obs
n_vars_aligned = len(genes_list)

# --- FIX 1: Initialize as LIL matrix for efficient block assignment ---
X_aligned = sp.lil_matrix((n_obs, n_vars_aligned), dtype=adata_eye.X.dtype)

adata_eye_aligned = ad.AnnData(
    X=X_aligned,
    obs=adata_eye.obs,
    var=pd.DataFrame(index=genes_list)
)

common_genes = list(set(adata_eye.var_names) & set(genes_list))
print(f"Number of common genes found: {len(common_genes)}")

gene_to_aligned_idx = {gene: i for i, gene in enumerate(adata_eye_aligned.var_names)}

chunk_size = 1000 # Increased chunk size for better performance

for i in tqdm(range(0, len(common_genes), chunk_size), desc="Processing blocks"):
    block_genes = common_genes[i:i + chunk_size]
    X_block = adata_eye[:, block_genes].X
    target_indices = [gene_to_aligned_idx[gene] for gene in block_genes]

    # This assignment is now fast and stable with a LIL matrix
    adata_eye_aligned.X[:, target_indices] = X_block

# --- FIX 2: Convert back to CSR for efficient analysis ---
print("Converting matrix to CSR format...")
adata_eye_aligned.X = adata_eye_aligned.X.tocsr()

print(f"Aligned dataset created with {adata_eye_aligned.n_vars} genes for {adata_eye_aligned.n_obs} cells.")
print(f"Found and copied data for {len(common_genes)} common genes.")

sc.pp.filter_cells(adata_eye_aligned, min_genes=n_genes_filter)

print(f"Filterd {adata_eye.n_obs - adata_eye_aligned.n_obs} on original total {adata_eye.n_obs}")

# save the aligned dataset
adata_eye_aligned.write_h5ad(f"{root_dir}/eye_sc_atlas_processed.h5ad")

X_eye = adata_eye_aligned.raw.X

max_val_eye = np.max(X_eye)

print(f"Max value in eye dataset: {max_val_eye}")