# scMulan_v1
Repository for paper scMulan: a multitask generative pre-trained language model for single-cell analysis (online soon).

scMulan is a foundation model for single cell transcriptomics. 

It is designed for zeroshot cell type annotation and batch integration, and conditional cell generation for in-silico perturbation.



# Installation
```
conda create -n scMulan python==3.10
conda activate scMulan
pip install -r requirements.txt
```

# Tutorials
We provided a tutorial of using scMulan for cell type annotation, in Tutorial-cell_type_annotation.ipynb
Currently, scMulan supports zero-shot annotation of human cell types in seven organs including Heart, Lung, Liver, Bone marrow, Blood, Brain, and Thymus.
It could also be used to get cell embeddings for batch integration.
You can easily use your adata and get annotaion from scMulan.
