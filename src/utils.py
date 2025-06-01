import numpy as np
import json


def load_dataset(dataset_name):
    dataset_path = f"datasets/{dataset_name}/{dataset_name}.json"
    with open(dataset_path, "r") as f:
        dataset = json.loads(f.read())
    return dataset

def rearrange_heads_by_layer(heads):
    arranged = {}
    for (layer,head) in heads:
        arranged.setdefault(layer,[])
        arranged[layer].append(head)
    return arranged

def get_w_vo(layer, head, cfg, state_dict, is_gqa):
    if is_gqa:
      n_attn_groups = cfg.n_heads // cfg.n_key_value_heads
      value_head = head // n_attn_groups
      w_v = state_dict[f"blocks.{layer}.attn._W_V"][value_head]
    else:
      w_v = state_dict[f"blocks.{layer}.attn.W_V"][head]
    w_o = state_dict[f"blocks.{layer}.attn.W_O"][head]
    w_vo = w_v @ w_o
    return w_vo

def top_k_indices(matrix, k):
    sorted_indices = np.argsort(matrix.flatten(), kind='stable')[::-1][:k]
    return np.array(np.unravel_index(sorted_indices, matrix.shape)).T

def top_k_indices_old(matrix, k):
    flat_indices = np.argpartition(matrix.ravel(), -k)[-k:]
    # Sort the top-k indices by value (descending order)
    top_k_sorted = flat_indices[np.argsort(matrix.ravel()[flat_indices], kind='stable')[::-1]]
    # Convert flat indices back to (i, j) indices
    return np.array(np.unravel_index(top_k_sorted, matrix.shape)).T

def get_k(model_name, relation_name):
    if "Llama-3" in model_name:
        k = 3 if "copying" in relation_name else 25
    else:
        k = 1 if "copying" in relation_name else 10
    return k

def get_topm_relation_heads(m, maps, dataset, apply_first_mlp, k, only_nonzero):
    relation_scores, _ = maps.calc_relation_scores(dataset, apply_first_mlp, k)
    top_k_heads = top_k_indices(relation_scores, m)
    if only_nonzero:
        top_k_heads = [(layer,head) for (layer,head) in top_k_heads if relation_scores[layer,head] > 0.0]
    return top_k_heads
