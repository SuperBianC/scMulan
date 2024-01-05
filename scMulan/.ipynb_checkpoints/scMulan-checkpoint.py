import torch
import os
import sys
from model.model import GPTConfig, cellGPTModel
from model.model_kvcache import cellGPTModel_kv
import torch.nn.functional as F
from utils.hf_tokenizer import cellGenesisTokenizer
import scipy.sparse
import numpy as np
from tqdm import tqdm

class scMulan:
    def __init__(self, ckp_path, adata, kv_cache = False, meta_info_path = os.path.join(os.path.dirname(__file__),'utils/meta_info.pt')):
        
        ckp = torch.load(ckp_path, map_location='cpu')
        gptconf = GPTConfig(**ckp['model_args'])
        if kv_cache:
            self.model = cellGPTModel_kv(gptconf)
        else:
            self.model = cellGPTModel(gptconf)
        self.model = self.model.cuda()
        self.model.load_state_dict(ckp['model'])
        self.model.eval()
        self.meta_info = torch.load(meta_info_path)
        self.tokenizer = cellGenesisTokenizer(self.meta_info['token_set'])
        self.n_express_level = ckp['model_args']['expression_level']
        self.check_adata(adata)
        self.mulan_gene_set = self.meta_info['gene_set']
        self.mulan_cell_type_entities = list(self.meta_info['cell_type'] | self.meta_info['MCT'])
        self.adata = adata

    def data_preprocess(self,):

        # use mulan gene set
        self.adata = self.adata[:,self.mulan_gene_set].copy()
        # sparse check
        self.adata_sparse = scipy.sparse.issparse(self.adata.X)
        # get COO matrix for analysis
        if self.adata_sparse:
            self.adata_matrix = self.adata.X.tocoo()
        else:
            print('adata is not sparse, use dense matrix')
            self.adata_matrix = self.adata.X


    def get_gene_expression_dict(self, i, matrix):
        # TODO what if it is not sparse
        if self.adata_sparse:
            cell_data = matrix.getrow(i).tocoo()
            cell_expression_dict = {self.mulan_gene_set[j]: cell_data.data[k] for k, j in enumerate(cell_data.col)}
            return cell_expression_dict

        cell_expression_dict = {self.mulan_gene_set[i]: expr for i, expr in enumerate(matrix[i]) if expr != 0}

        return cell_expression_dict
    
    def prepare_gene_expression_codings(self, i, matrix):

        cell_expression_dict = self.get_gene_expression_dict(i, matrix)
        expressed_genes = list(cell_expression_dict.keys())[::-1]
        expression_values = list(cell_expression_dict.values())[::-1]
        max_expression = np.max(expression_values)
        bins = np.linspace(0, max_expression, self.n_express_level+1)
        binned_expr = np.digitize(expression_values, bins, right=True)

        return expressed_genes, binned_expr
    
    def make_encoded_annotation_prompt_one_cell(self, expressed_genes, binned_expr, annotation_task_token = '<PCT>'):

        prefix = expressed_genes + [annotation_task_token] # add pre-defined task token to guide model generate cell type annotations
        ec_binned_expr = np.append(binned_expr,[0]*(len([annotation_task_token]))) # add a zero for task token
        ec_prefix = self.tokenizer.encode(prefix) 
        prefix_len_with_task_token = len(ec_prefix) # length with task token

        return (ec_prefix, ec_binned_expr, prefix_len_with_task_token)
    
    def is_cell_type_entity(self, token_entity):
        return token_entity in self.mulan_cell_type_entities
    
    def predict_cell_type_one_cell(self, i, matrix, ignore_Idx = None):

        expressed_genes, binned_expr = self.prepare_gene_expression_codings(i, matrix)
        ec_prefix, ec_binned_expr, prefix_len_with_task_token = self.make_encoded_annotation_prompt_one_cell(expressed_genes, binned_expr)
        prompt_entities = torch.tensor(ec_prefix[:prefix_len_with_task_token]).unsqueeze(0).cuda()
        prompt_values = torch.tensor(ec_binned_expr[:prefix_len_with_task_token]).unsqueeze(0).cuda()
        with torch.no_grad():
            generated_tokens = self.model.generate_cellGenesis(prompt_entities,prompt_values, max_new_tokens= prefix_len_with_task_token + 3,ignore_Idx=ignore_Idx,top_k=1)[0].cpu().tolist()
        pred_names = self.tokenizer.convert_ids_to_tokens(generated_tokens[0][-3:-1])
        coarse_cell_type = pred_names[-2] if self.is_cell_type_entity(pred_names[-2]) else 'Unclassified'
        fine_cell_type = pred_names[-1] if self.is_cell_type_entity(pred_names[-1]) else 'Unclassified'

        return coarse_cell_type, fine_cell_type

    def get_cell_embedding(self, i, matrix, ignore_Idx = None):

        expressed_genes, binned_expr = self.prepare_gene_expression_codings(i, matrix)
        ec_prefix, ec_binned_expr, prefix_len_with_task_token = self.make_encoded_annotation_prompt_one_cell(expressed_genes, binned_expr)
        prompt_entities = torch.tensor(ec_prefix[:prefix_len_with_task_token]).unsqueeze(0).cuda()
        prompt_values = torch.tensor(ec_binned_expr[:prefix_len_with_task_token]).unsqueeze(0).cuda()
        generated_entities, generated_values, hidden = self.model.generate_cellGenesis(prompt_entities,prompt_values, max_new_tokens= prefix_len_with_task_token + 3,ignore_Idx=ignore_Idx,top_k=1, return_hidden=True) # +3 is passing CT1, CT2,<#E#>

        return hidden[-1].cpu().detach()


    def get_cell_embedding_for_adata(self, ignore_Idx = None):

        self.data_preprocess()
        hidden_states = []
        for i in tqdm(range(self.adata.n_obs), desc="⏳Collecting cell embeddings"):
            hidden = self.get_cell_embedding(i, self.adata_matrix)
            hidden_states.append(hidden)
            
        return hidden_states

    def predict_cell_types_for_adata(self, ignore_Idx = None):

        self.data_preprocess()
        fine_cell_type_pred = []
        for i in tqdm(range(self.adata.n_obs), desc="⏳Generating cell type labels:"):
            _, fine_cell_type = self.predict_cell_type_one_cell(i, self.adata_matrix)
            fine_cell_type_pred.append(fine_cell_type)
        self.adata.obs['cell_type_from_mulan'] = fine_cell_type_pred

        return self.adata
        


    def check_adata(self, adata):
        # check normalize and log1p
        adata_max = adata.X.max()
        assert adata_max < 10, f'🚫 Please make sure adata is processed with normalization (sum = 1e4) and log1p, your adata max is {adata_max}.'
        # check gene symbol uniform
        adata_var = set(adata.var_names.tolist())
        mulan_geneset = set(self.meta_info['gene_set'])
        count = len(adata_var.intersection(mulan_geneset))
        assert count == len(self.meta_info['gene_set']), f'🚫 Please make sure adata is processed with uniformed gene symbol, your gene set has {count} overlap with scMulan.'
        print('✅ adata passed check')
        print("⚔️ scMulan is ready")

    


        

    



    

        