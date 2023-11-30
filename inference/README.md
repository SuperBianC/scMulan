# inference

## input
h5ad file for annotation or terms for cell generation

## process

normalize to 10000 and log1p
select genes from scMulan's gene set
transfer it into h5
decide what task you want to conduct
transform gene expression matrix into c-sentence
combine c-sentence with tasks
input the combined c-sentence into scMulan sentence by sentence (We recommend use multi-GPUs for parallel (e.g. four 3090-GPU or orther verison with more FLOPs))

# Tutorials Coming soon

