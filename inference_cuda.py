import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" ## set your available devices, each use ~2G GPU-MEMORY
import scanpy as sc
import scMulan
from scMulan import GeneSymbolUniform
adata = sc.read('Data/liver.h5ad', backup_url='https://cloud.tsinghua.edu.cn/f/45a7fd2a27e543539f59/?dl=1')
adata_GS_uniformed = GeneSymbolUniform(input_adata=adata,
                                 output_dir="Data/",
                                 output_prefix='liver')
                                 # norm and log1p count matrix
if adata_GS_uniformed.X.max() > 10:
    sc.pp.normalize_total(adata_GS_uniformed, target_sum=1e4) 
    sc.pp.log1p(adata_GS_uniformed)

# you should first download ckpt from https://cloud.tsinghua.edu.cn/f/2250c5df51034b2e9a85/?dl=1
# put it under .ckpt/ckpt_scMulan.pt
# by: wget https://cloud.tsinghua.edu.cn/f/2250c5df51034b2e9a85/?dl=1  -O ckpt/ckpt_scMulan.pt

ckp_path = 'ckpt/ckpt_scMulan.pt'
scml = scMulan.model_inference(ckp_path, adata_GS_uniformed)
scml.get_cell_types_and_embds_for_adata(parallel=False)
adata_mulan = scml.adata.copy()
