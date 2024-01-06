# scMulan_v1

Welcome to the repository for scMulan_v1, featuring our upcoming work: "scMulan: A Multitask Generative Pre-trained Language Model for Single-Cell Analysis." 



## Introducing scMulan 
scMulan is a groundbreaking foundation model for the analysis of single-cell transcriptomics.
Features:
- **Zero-shot Cell Type Annotation:** Effortlessly classify cell types without prior training or reference mapping.
- **Zero-shot Batch Integration:** Seamlessly integrate data from diverse batches.
- **Conditional Cell Generation:** Support simulation of in-silico perturbations.

## Installation
```
conda create -n scMulan python==3.10
conda activate scMulan
pip install -r requirements.txt
```

## Quick start
🚀 Jump right into analysis! All you need is your test adata file.

## Tutorials
We provided a tutorial of using scMulan for [cell type annotation](/Tutorial-cell_type_annotation.ipynb).
Currently, scMulan supports zero-shot annotation of human cell types in seven organs including Heart, Lung, Liver, Bone marrow, Blood, Brain, and Thymus.

It could also be used to get cell embeddings for [batch integration](/Tutorial-integration.ipynb).
You can easily use your adata and get analysis from scMulan.
