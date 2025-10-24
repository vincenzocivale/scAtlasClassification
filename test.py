import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
from scipy import sparse as sp
from tqdm import tqdm
import gc

# Configuration
genes_list = pd.read_csv("OS_scRNA_gene_index.19264.tsv", header=None, sep="\t")[0].tolist()
label_obs = 'cell_type'
n_genes_filter = 200
root_dir = "/equilibrium/datasets/TCGA-histological-data/scDataset"
obs_chunk_size = 5000  # Process 5000 cells at a time - adjust based on available RAM

print("Loading dataset metadata...")
adata_eye = sc.read_h5ad(f"{root_dir}/nervous_system_sc_atlas.h5ad", backed='r')

# Assign gene names and make unique
adata_eye.var_names = adata_eye.var['feature_name'].astype(str)
adata_eye.var_names_make_unique()

print(f"Dataset shape: {adata_eye.n_obs} cells × {adata_eye.n_vars} genes")

# Find common genes and create index mapping
common_genes = list(set(adata_eye.var_names) & set(genes_list))
print(f"Number of common genes found: {len(common_genes)}/{len(genes_list)}")

# Create mapping: gene name -> index in original adata
gene_to_original_idx = {gene: i for i, gene in enumerate(adata_eye.var_names)}
original_indices = [gene_to_original_idx[gene] for gene in common_genes if gene in gene_to_original_idx]

# Create mapping: gene name -> index in aligned adata
gene_to_aligned_idx = {gene: i for i, gene in enumerate(genes_list)}
aligned_indices = [gene_to_aligned_idx[gene] for gene in common_genes if gene in gene_to_aligned_idx]

print(f"Verified {len(original_indices)} genes can be mapped")

# Initialize storage for aligned data chunks
n_obs = adata_eye.n_obs
n_vars_aligned = len(genes_list)
aligned_chunks = []
obs_list = []

print("Processing data in cell chunks...")
n_chunks = (n_obs + obs_chunk_size - 1) // obs_chunk_size

for chunk_idx in tqdm(range(n_chunks), desc="Processing cell chunks"):
    start_idx = chunk_idx * obs_chunk_size
    end_idx = min(start_idx + obs_chunk_size, n_obs)
    
    # Load chunk of observations with only the genes we need
    X_chunk_subset = adata_eye[start_idx:end_idx, original_indices].X
    
    # Convert to dense if sparse (for easier manipulation of small chunks)
    if sp.issparse(X_chunk_subset):
        X_chunk_subset = X_chunk_subset.toarray()
    
    # Create empty aligned matrix for this chunk
    X_chunk_aligned = np.zeros((end_idx - start_idx, n_vars_aligned), dtype=X_chunk_subset.dtype)
    
    # Fill in the common genes at their aligned positions
    X_chunk_aligned[:, aligned_indices] = X_chunk_subset
    
    # Convert back to sparse and append
    aligned_chunks.append(sp.csr_matrix(X_chunk_aligned))
    obs_list.append(adata_eye.obs.iloc[start_idx:end_idx])
    
    # Clear memory
    del X_chunk_subset, X_chunk_aligned
    gc.collect()

print("Concatenating chunks...")
X_aligned = sp.vstack(aligned_chunks)
obs_aligned = pd.concat(obs_list, axis=0)

# Clear chunk storage
del aligned_chunks, obs_list
gc.collect()

print("Creating aligned AnnData object...")
adata_eye_aligned = ad.AnnData(
    X=X_aligned,
    obs=obs_aligned,
    var=pd.DataFrame(index=genes_list)
)

print(f"Aligned dataset created with {adata_eye_aligned.n_vars} genes for {adata_eye_aligned.n_obs} cells.")

# Filter cells
print(f"Filtering cells with < {n_genes_filter} genes...")
n_obs_before = adata_eye_aligned.n_obs
sc.pp.filter_cells(adata_eye_aligned, min_genes=n_genes_filter)
print(f"Filtered {n_obs_before - adata_eye_aligned.n_obs} cells from original total {n_obs_before}")

# Save the aligned dataset
print("Saving aligned dataset...")
adata_eye_aligned.write_h5ad(f"{root_dir}/nervous_system_sc_atlas_processed.h5ad")

# Get max value
if hasattr(adata_eye_aligned, 'raw') and adata_eye_aligned.raw is not None:
    X_eye = adata_eye_aligned.raw.X
else:
    X_eye = adata_eye_aligned.X

if sp.issparse(X_eye):
    max_val_eye = X_eye.max()
else:
    max_val_eye = np.max(X_eye)

print(f"Max value in aligned dataset: {max_val_eye}")
print("Done!")