import os
import torch
from transformers import AutoTokenizer
import transformer_lens
torch.set_default_device("cuda")
import sys
import pandas as pd
from tqdm import tqdm
from src.utils import get_k, load_dataset
from src.maps import MAPS
from src.experiment1_correlative import CorrelativeExperiment
from datetime import datetime



RELATIONS = [
    "country_to_capital_wikidata",
    "country_to_official_language_wikidata",
    "entity_to_pronoun",
    "general_copying_english_500",
    "name_copying",
    "object_superclass",
    "word_first_letter",
    "word_last_letter",
    "word_to_synonym",
    "work_location",
    "year_to_following"
]

def load_and_prepare_dataset(relation_name, tokenizer):
    dataset = load_dataset(relation_name)
    filtered_dataset = []
    srcs = []
    dsts = []
    for (src,dst) in dataset:
        dst_tokenized = tokenizer.encode(" "+dst, add_special_tokens=False)
        if len(dst_tokenized) != 1:
            continue
        src_tokenized = tokenizer.encode(src, add_special_tokens=False)
        if len(src_tokenized) == 1:
            continue
        filtered_dataset.append((src,dst))
        srcs.append(src_tokenized)
        dsts.append(dst_tokenized)
    return dataset, filtered_dataset, srcs, dsts

def get_scores_one_relation(maps: MAPS, 
                            relation_name, 
                            tokenizer, 
                            tokenized_template, 
                            k, 
                            src_token_position):
    dataset, filtered_dataset, srcs, dsts = load_and_prepare_dataset(relation_name, tokenizer)
    dynamic_scores = None
    for ix in range(len(filtered_dataset)):
        dst_tokens = torch.tensor(dsts[ix])
        tokenized_prompt = tokenized_template + srcs[ix]
        tokenized_prompt = torch.tensor(tokenized_prompt).unsqueeze(0)
        prompt_dynamic_scores = maps.calc_dynamic_relation_scores_from_prompts(tokenized_prompt, src_token_position, k, dst_tokens)
        if not dynamic_scores:
            dynamic_scores = prompt_dynamic_scores
        else:
            for key in prompt_dynamic_scores:
                dynamic_scores[key] += prompt_dynamic_scores[key]
    for key in dynamic_scores:
        dynamic_scores[key] /= len(filtered_dataset)
    return dynamic_scores

def save_dynamic_scores(dynamic_scores, model_name, relation_name):
    dynamic_scores_ = {key:dynamic_scores[key].flatten() for key in dynamic_scores}
    df = pd.DataFrame(dynamic_scores_)
    path = f"results/{model_name}/multi_token/dynamic_scores/{relation_name}"
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path,"results.csv")
    df.to_csv(path,index=False)

def run_multi_token_experiment(maps:MAPS, tokenizer, model_name, cfg, model):
    template = "This is a document about <X>"
    tokenized_template = tokenizer(template.replace("<X>","")).input_ids
    src_token_position = -1
    results_dir = f"results/{model_name}/multi_token"
    corr_exp = CorrelativeExperiment(MAPS(model,tokenizer), model_name, "", cfg, n_templates=1, results_dir=results_dir)
    summary_list = []
    for relation_name in tqdm(RELATIONS):
        if not MAPS(model,tokenizer).is_valid_dataset_for_model(load_dataset(relation_name)):
            print(f"{relation_name} is not a valid dataset for {model_name}",flush=True)
            continue
        else:
            print(f"{relation_name} is a valid dataset for {model_name}",flush=True)
        k = get_k(model_name, relation_name)
        dynamic_scores = get_scores_one_relation(maps, 
                                    relation_name, 
                                    tokenizer, 
                                    tokenized_template, 
                                    k, 
                                    src_token_position)
        save_dynamic_scores(dynamic_scores, model_name, relation_name)
        relation_scores, suppression_relation_scores = corr_exp.get_static_scores(relation_name, load_dataset(relation_name))
        correlations = corr_exp.calc_correlations(relation_name, relation_scores, suppression_relation_scores, dynamic_scores)
        summary_list.append(correlations)
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv(os.path.join(corr_exp.correlations_output_dir, "summary.csv"), index=False)
    corr_exp.summary_df_to_latex(summary_df)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Usage: python multi_token.py <model_name>", flush=True)
        sys.exit(1)
    model_name = sys.argv[1]
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] running multi-token experiment for model: {model_name}",flush=True)
    model = transformer_lens.HookedTransformer.from_pretrained_no_processing(model_name, device_map="auto")
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    cfg = model.cfg
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    last_device = torch.device(f"cuda:{num_gpus-1}" if torch.cuda.is_available() else "cpu")
    maps = MAPS(model, tokenizer, add_leading_space=False, min_filtered_dataset_size=30)
    run_multi_token_experiment(maps, tokenizer, model_name, cfg, model)