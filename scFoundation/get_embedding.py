# Copyright 2023 BioMap (Beijing) Intelligence Technology Limited

import argparse
import random,os
import numpy as np
import pandas as pd
import argparse
import torch
from tqdm import tqdm
import scipy.sparse
from scipy.sparse import issparse
import scanpy as sc
from load import *
import time



####################################Settings#################################
parser = argparse.ArgumentParser(description='Drug_response_pre')
parser.add_argument('--task_name', type=str, default='deepcdr', help='task name')
parser.add_argument('--input_type', type=str, default='singlecell',choices=['singlecell','bulk'], help='input type; default: singlecell')
parser.add_argument('--output_type', type=str, default='cell',choices=['cell','gene','gene_batch','gene_expression'], help='cell or gene embedding; default: cell the difference between gene and gene_batch is that in gene mode the gene embedding will be processed one by one. while in gene_batch mode, the gene embedding will be processed in batch. GEARS use gene_batch mode.')
parser.add_argument('--pool_type', type=str, default='all',choices=['all','max'], help='pooling type of cell embedding; default: all only valid for output_type=cell')
parser.add_argument('--tgthighres', type=str, default='t4', help='the targeted high resolution (start with t) or the fold change of the high resolution (start with f), or the addtion (start with a) of the high resoultion. only valid for input_type=singlecell')
parser.add_argument('--data_path', type=str, default='./', help='input data path')
parser.add_argument('--save_path', type=str, default='./', help='save path')
parser.add_argument('--pre_normalized', type=str, default='F',choices=['F','T','A'], help='if normalized before input; default: False (F). choice: True(T), Append(A) When input_type=bulk: pre_normalized=T means log10(sum of gene expression). pre_normalized=F means sum of gene expression without normalization. When input_type=singlecell: pre_normalized=T or F means gene expression is already normalized+log1p or not. pre_normalized=A means gene expression is normalized and log1p transformed. the total count is appended to the end of the gene expression matrix.')
parser.add_argument('--demo', action='store_true', default=False, help='if demo, only infer 10 samples')
parser.add_argument('--version',  type=str, default='ce', help='only valid for output_type=cell. For read depth enhancemnet, version=rde For others, version=ce')
parser.add_argument('--model_path',  type=str, default='None', help='pre-trained model path')
parser.add_argument('--ckpt_name',  type=str, default='01B-resolution', help='checkpoint name')
# --- NUOVO ---
# Aggiunto un argomento per il filtraggio delle cellule
parser.add_argument('--min_genes_cell', type=int, default=1, help='Filter out cells with fewer genes than this number. Default: 1, to remove empty cells.')
# --- FINE NUOVO ---


args = parser.parse_args()



def main_gene_selection(X_df, gene_list):
    """
    Describe:
        rebuild the input adata to select target genes encode protein 
    Parameters:
        adata->`~anndata.AnnData` object: adata with var index_name by gene symbol
        gene_list->list: wanted target gene 
    Returns:
        adata_new->`~anndata.AnnData` object
        to_fill_columns->list: zero padding gene
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))), 
                                 columns=to_fill_columns, 
                                 index=X_df.index)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                        index=X_df.index, 
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns,var

gene_list_df = pd.read_csv('./OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
gene_list = list(gene_list_df['gene_name'])

def main():
    # Set random seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # --- CARICAMENTO DATI ---
    if args.data_path.endswith('npz'):
        gexpr_feature = scipy.sparse.load_npz(args.data_path).toarray()
        gexpr_feature = pd.DataFrame(gexpr_feature)
    elif args.data_path.endswith('h5ad'):
        adata = sc.read_h5ad(args.data_path)
        if args.min_genes_cell > 0:
            print(f"Original number of cells: {adata.n_obs}")
            sc.pp.filter_cells(adata, min_genes=args.min_genes_cell)
            print(f"Number of cells after filtering ({args.min_genes_cell} min genes): {adata.n_obs}")
        idx = adata.obs_names.tolist()
        try:
            col = adata.var.gene_name.tolist()
        except:
            col = adata.var_names.tolist()
        gexpr_feature = adata.X.toarray() if issparse(adata.X) else adata.X
        gexpr_feature = pd.DataFrame(gexpr_feature, index=idx, columns=col)
    elif args.data_path.endswith('npy'):
        gexpr_feature = pd.DataFrame(np.load(args.data_path))
    else:
        gexpr_feature = pd.read_csv(args.data_path, index_col=0)
    
    if gexpr_feature.shape[1] < 19264:
        print('Convert gene feature into 19264...')
        gexpr_feature, to_fill_columns, var = main_gene_selection(gexpr_feature, gene_list)

    if (args.pre_normalized == 'F') and (args.input_type == 'bulk'):
        adata = sc.AnnData(gexpr_feature)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        gexpr_feature = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)

    if args.demo:
        gexpr_feature = gexpr_feature.iloc[:10, :]

    print(f"Dataset shape: {gexpr_feature.shape}")

    # --- MODELLO ---
    if args.version == 'noversion':
        ckpt_path = args.model_path
        key = None
    else:
        ckpt_path = args.model_path
        if args.output_type == 'cell':
            key = 'cell' if args.version == 'ce' else 'rde'
        else:
            key = 'gene'

    print(f"Load model from {ckpt_path}")
    pretrainmodel, pretrainconfig = load_model_frommmf(ckpt_path, key)
    pretrainmodel.eval()

    # --- PARAMETRI SALVATAGGIO ---
    SAVE_EVERY = 1000  # salva ogni N campioni
    strname = os.path.join(args.save_path, f"{args.task_name}_{args.ckpt_name}_{args.input_type}_{args.output_type}_embedding_{args.tgthighres}_resolution.npy")
    temp_save = strname.replace('.npy', '_partial.npy')
    progress_file = strname.replace('.npy', '_progress.txt')
    print(f"Salverò progressivamente ogni {SAVE_EVERY} campioni in: {temp_save}")

    # --- RIPRESA AUTOMATICA ---
    start_index = 0
    geneexpemb = []

    if os.path.exists(temp_save) and os.path.exists(progress_file):
        print("🔁 Rilevato checkpoint parziale. Ripristino in corso...")
        geneexpemb = list(np.load(temp_save, allow_pickle=True))
        with open(progress_file, 'r') as f:
            start_index = int(f.read().strip())
        print(f"→ Riprendo da campione {start_index}.")

    total_inference_time = 0.0

    # --- INFERENZA ---
    for i in tqdm(range(start_index, gexpr_feature.shape[0])):
        start_time = time.time()
        with torch.no_grad():
            # --- BLOCCO DATI ---
            if args.input_type == 'singlecell':
                if args.pre_normalized == 'F':
                    current_sum = gexpr_feature.iloc[i, :].sum()
                    tmpdata = np.zeros_like(gexpr_feature.iloc[i, :]) if current_sum == 0 else np.log1p(gexpr_feature.iloc[i, :] / current_sum * 1e4)
                elif args.pre_normalized == 'T':
                    tmpdata = gexpr_feature.iloc[i, :]
                elif args.pre_normalized == 'A':
                    tmpdata = gexpr_feature.iloc[i, :-1]
                else:
                    raise ValueError('pre_normalized must be T, F or A')

                totalcount = gexpr_feature.iloc[i, -1] if args.pre_normalized == 'A' else gexpr_feature.iloc[i, :].sum()
                log_totalcount = np.log10(totalcount + 1e-10)

                if args.tgthighres[0] == 'f':
                    pretrain_gene_x = torch.tensor(list(tmpdata) + [np.log10(totalcount * float(args.tgthighres[1:]) + 1e-10), log_totalcount]).unsqueeze(0).to(device)
                elif args.tgthighres[0] == 'a':
                    pretrain_gene_x = torch.tensor(list(tmpdata) + [log_totalcount + float(args.tgthighres[1:]), log_totalcount]).unsqueeze(0).to(device)
                elif args.tgthighres[0] == 't':
                    pretrain_gene_x = torch.tensor(list(tmpdata) + [float(args.tgthighres[1:]), log_totalcount]).unsqueeze(0).to(device)
                else:
                    raise ValueError('tgthighres must start with f, a or t')

                data_gene_ids = torch.arange(19266, device=device).repeat(1, 1)
            else:
                raise ValueError('Questo script è pensato per input_type=singlecell')

            # --- EMBEDDING ---
            value_labels = pretrain_gene_x > 0
            x, x_padding = gatherData(pretrain_gene_x, value_labels, pretrainconfig['pad_token_id'])
            if args.output_type == 'cell':
                position_gene_ids, _ = gatherData(data_gene_ids, value_labels, pretrainconfig['pad_token_id'])
                x = pretrainmodel.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
                position_emb = pretrainmodel.pos_emb(position_gene_ids)
                x += position_emb
                geneemb = pretrainmodel.encoder(x, x_padding)

                geneemb1 = geneemb[:, -1, :]
                geneemb2 = geneemb[:, -2, :]
                geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)
                geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)
                if args.pool_type == 'all':
                    geneembmerge = torch.concat([geneemb1, geneemb2, geneemb3, geneemb4], axis=1)
                elif args.pool_type == 'max':
                    geneembmerge, _ = torch.max(geneemb, dim=1)
                else:
                    raise ValueError('pool_type must be all or max')
                geneexpemb.append(geneembmerge.detach().cpu().numpy())

        end_time = time.time()
        total_inference_time += (end_time - start_time)

        # --- SALVATAGGIO PROGRESSIVO ---
        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == gexpr_feature.shape[0]:
            np.save(temp_save, np.array(geneexpemb, dtype=object))
            with open(progress_file, 'w') as f:
                f.write(str(i + 1))
            print(f"💾 Checkpoint salvato a {i + 1} campioni ({len(geneexpemb)} embedding).")

    # --- SALVATAGGIO FINALE ---
    geneexpemb = np.squeeze(np.array(geneexpemb))
    np.save(strname, geneexpemb)
    print(f"\n✅ Salvataggio finale completato: {strname}")

    # --- PULIZIA ---
    if os.path.exists(temp_save):
        os.remove(temp_save)
    if os.path.exists(progress_file):
        os.remove(progress_file)

    avg_time = total_inference_time / gexpr_feature.shape[0]
    print(f"\nTempo medio di inferenza per cellula: {avg_time:.4f} secondi")
    print(f"Tempo totale di inferenza: {total_inference_time:.2f} secondi")
 

if __name__=='__main__':
    main()