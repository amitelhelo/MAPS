import os
import torch
import numpy as np
torch.set_default_device("cuda")
from functools import partial
import pandas as pd
from src.utils import get_w_vo, rearrange_heads_by_layer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
last_device = torch.device(f"cuda:{torch.cuda.device_count()-1}" if torch.cuda.is_available() else "cpu")


class MAPS:

    def __init__(self, model, tokenizer, add_leading_space=True, min_filtered_dataset_size=30):
        self.tokenizer = tokenizer
        self.model = model
        self.state_dict = model.state_dict()
        self.first_mlp = lambda x: self.model.blocks[0].mlp.to(device)(model.blocks[0].ln2.to(device)(x))
        self.cfg = model.cfg
        self.is_gqa = self.cfg.n_key_value_heads != None
        self.add_leading_space = add_leading_space
        self.min_filtered_dataset_size = min_filtered_dataset_size

    # input: list of (src,dst) pairs - e.g. [("France","Paris")]
    def preprocess_dataset(self, dataset, filter_dataset=True):
        if self.add_leading_space:
            dataset_ = [(" "+x, " "+y) for (x,y) in dataset]
        else:
            dataset_ = dataset
        tokenized_dataset = [(self.tokenizer.encode(x, add_special_tokens=False), self.tokenizer.encode(y, add_special_tokens=False)) for (x,y) in dataset_]
        if filter_dataset:
            filtered_dataset = [dataset[i] for i in range(len(dataset)) if (len(tokenized_dataset[i][0])==len(tokenized_dataset[i][1])==1)]
            filtered_tokenized_dataset = [tokenized_dataset[i] for i in range(len(dataset)) if (len(tokenized_dataset[i][0])==len(tokenized_dataset[i][1])==1)]
        else:
            filtered_dataset = dataset
            filtered_tokenized_dataset = tokenized_dataset 
        src_tokens = torch.tensor([x[0] for (x,y) in filtered_tokenized_dataset])
        dst_tokens = torch.tensor([y[0] for (x,y) in filtered_tokenized_dataset])
        return filtered_dataset, src_tokens, dst_tokens

    def is_valid_dataset_for_model(self, dataset):
        filtered_dataset, _, _ = self.preprocess_dataset(dataset)
        return len(filtered_dataset) >= self.min_filtered_dataset_size

    def ov_project(self, src_embeds, layer, head):
        W_VO = get_w_vo(layer, head, self.cfg, self.state_dict, self.is_gqa)
        transformed = src_embeds @ W_VO
        projected = transformed @ self.state_dict["unembed.W_U"]
        return projected

    def calc_relation_score(self, projection, k, dst_tokens):
        dst_tokens = dst_tokens.to(projection.device)
        _, indices = torch.topk(projection, k)
        matches = (indices == dst_tokens.unsqueeze(1))
        matches = matches.sum(dim=1)
        relation_score = matches.float().mean().item()
        return relation_score

    def calc_relation_scores(self, dataset, apply_first_mlp, k):
        _, src_tokens, dst_tokens = self.preprocess_dataset(dataset)
        relation_scores = np.zeros((self.cfg.n_layers, self.cfg.n_heads))
        suppression_relation_scores = np.zeros((self.cfg.n_layers, self.cfg.n_heads))
        src_embeds = self.state_dict["embed.W_E"][src_tokens]
        if apply_first_mlp:
            src_embeds_ = self.first_mlp(src_embeds)
        else:
            src_embeds_ = src_embeds
        for layer in range(self.cfg.n_layers):
            for head in range(self.cfg.n_heads):
                if layer == 0:
                    projection = self.ov_project(src_embeds, layer, head)
                else:
                    projection = self.ov_project(src_embeds_, layer, head)
                relation_scores[layer,head] = self.calc_relation_score(projection, k, dst_tokens)
                suppression_relation_scores[layer,head] = self.calc_relation_score(-projection, k, dst_tokens)
        return relation_scores, suppression_relation_scores
    
    def calc_dynamic_relation_scores_from_prompts(self, input_ids, src_token_position, k, dst_tokens):
        clean_run_cache = {}
        self.model.remove_all_hook_fns()
        self.model.add_caching_hooks([f"blocks.{layer}.attn.hook_z" for layer in range(self.cfg.n_layers)], cache=clean_run_cache)
        self.model(input_ids)
        self.model.remove_all_hook_fns()
        w_context_dynamic_relation_scores, w_context_suppression_dynamic_relation_scores = \
                    self.calc_dynamic_relation_score_from_activations(clean_run_cache ,k, dst_tokens)
        attn_fix_cache = {}
        def attn_fix_hook(attn_weights, hook):
            attn_weights[:,:,-1,:] = 0.0
            attn_weights[:,:,-1,src_token_position] = 1.0
        def plug_clean_activations_hook(activations, hook):
            attn_fix_cache[hook.name] = activations.clone()
            return clean_run_cache[hook.name]
        for layer in range(self.cfg.n_layers):
            self.model.add_hook(f"blocks.{layer}.attn.hook_pattern", attn_fix_hook)
            self.model.add_hook(f"blocks.{layer}.attn.hook_z", plug_clean_activations_hook)
        self.model(input_ids)
        self.model.remove_all_hook_fns()
        del clean_run_cache
        torch.cuda.empty_cache()
        wo_context_dynamic_relation_scores, wo_context_suppression_dynamic_relation_scores = \
                        self.calc_dynamic_relation_score_from_activations(attn_fix_cache ,k, dst_tokens)
        dynamic_results = {
            "w_context_dynamic_relation_scores": w_context_dynamic_relation_scores,
            "w_context_suppression_dynamic_relation_scores": w_context_suppression_dynamic_relation_scores,
            "wo_context_dynamic_relation_scores": wo_context_dynamic_relation_scores,
            "wo_context_suppression_dynamic_relation_scores": wo_context_suppression_dynamic_relation_scores
        }
        del attn_fix_cache
        torch.cuda.empty_cache()
        return dynamic_results

    def calc_dynamic_relation_scores(self, dataset, template ,k):
        filtered_dataset,src_tokens,dst_tokens = self.preprocess_dataset(dataset)
        prompts = [template.replace("<X>", src) for (src,_) in filtered_dataset]
        input_ids = self.tokenizer(prompts, return_tensors="pt").input_ids
        src_token_position = torch.where(input_ids[0]==src_tokens[0])[0].item()
        dynamic_results = self.calc_dynamic_relation_scores_from_prompts(input_ids, src_token_position, k, dst_tokens)
        return dynamic_results
    
    def calc_dynamic_relation_score_from_activations(self, cache, k, dst_tokens):
        relation_scores = np.zeros((self.cfg.n_layers, self.cfg.n_heads))
        suppression_relation_scores = np.zeros((self.cfg.n_layers, self.cfg.n_heads))
        for layer in range(self.cfg.n_layers):
            layer_output = cache[f"blocks.{layer}.attn.hook_z"][:,-1,:,:].to(last_device)
            heads_outputs = torch.einsum('abc,bcd->bad', layer_output, self.state_dict[f"blocks.{layer}.attn.W_O"].to(last_device))
            heads_projections = heads_outputs @ self.state_dict["unembed.W_U"].to(last_device)
            for head in range(self.cfg.n_heads):
                projection = heads_projections[head]
                relation_scores[layer,head] = self.calc_relation_score(projection, k, dst_tokens)
                suppression_relation_scores[layer,head] = self.calc_relation_score(-projection, k, dst_tokens)
        return relation_scores, suppression_relation_scores

    def calc_causal_effects(self, dataset, template, heads_to_ablate):
        def ablate_heads_in_layer_hook(activations, hook, heads):
            for head in heads:
                activations[:,-1,head,:] = 0.0
        self.model.remove_all_hook_fns()
        filtered_dataset,src_tokens,dst_tokens = self.preprocess_dataset(dataset)
        prompts = [template.replace("<X>", src) for (src,_) in filtered_dataset]
        input_ids = self.tokenizer(prompts, return_tensors="pt").input_ids
        def calc_accuracy(model):
            logits = model(input_ids)[:,-1,:]
            probs = torch.nn.functional.softmax(logits,dim=-1)
            next_tokens = torch.argmax(probs,dim=-1)
            accuracy = (next_tokens==dst_tokens).float().mean().item()
            return accuracy
        accuracies = [calc_accuracy(self.model)]
        current_heads_to_ablate = []
        for head in heads_to_ablate:
            current_heads_to_ablate.append(head)
            arranged_current_heads_to_ablate = rearrange_heads_by_layer(current_heads_to_ablate)
            for layer in arranged_current_heads_to_ablate:
                self.model.add_hook(f"blocks.{layer}.attn.hook_z", partial(ablate_heads_in_layer_hook, heads=arranged_current_heads_to_ablate[layer]))
            accuracies.append(calc_accuracy(self.model))
            self.model.remove_all_hook_fns()
        return accuracies

    def get_salient_operations(self, layer, head, k_salient_tokens, k_mappings, apply_first_mlp):
        W_VO = get_w_vo(layer, head, self.cfg, self.state_dict, self.is_gqa)
        src_embeds = self.state_dict["embed.W_E"]
        if apply_first_mlp:  # note: in the paper we (accidentally) applied the first mlp to the first layer as well
            src_embeds = self.first_mlp(src_embeds)
        after_transformation = src_embeds @ W_VO
        norms_amplification = (after_transformation.norm(dim=-1)) / (src_embeds.norm(dim=-1))
        salient_tokens_ids = torch.topk(norms_amplification, k_salient_tokens).indices
        salient_tokens_decoded = [repr(self.tokenizer.decode(tok)) for tok in salient_tokens_ids]
        salient_tokens_transformed = after_transformation[salient_tokens_ids]
        salient_tokens_projected = salient_tokens_transformed @ self.state_dict["unembed.W_U"]
        salient_mappings = torch.topk(salient_tokens_projected, k_mappings, dim=-1).indices
        salient_mappings_decoded = [
            [repr(self.tokenizer.decode(tok)) for tok in mappings_lst] for mappings_lst in salient_mappings
        ]
        return salient_tokens_decoded, salient_mappings_decoded