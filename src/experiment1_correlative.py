import os
import torch
from transformers import AutoTokenizer
import numpy as np
import transformer_lens
torch.set_default_device("cuda")
import pandas as pd
from matplotlib import pyplot as plt
import json
from tqdm import tqdm
import torch.nn.functional as F
from scipy.stats import pearsonr
from src.utils import get_k, load_dataset
from src.paper_utils import category_to_category_name, relation_to_category, format_output_df, relation_to_fixed_name, load_relation_scores
from src.maps import MAPS
import sys
from datetime import datetime

TEMPLATES = [
    "No <X> means no", 
    "The story of <X> contains", 
    "This is a document about <X>", 
    "When I think about <X> I think about"
    ]

class CorrelativeExperiment:

    def __init__(self, maps: MAPS, model_name, experiment_name, cfg, n_templates, results_dir):
        self.model_name = model_name
        self.maps = maps
        self.experiment_name = experiment_name
        self.cfg = cfg
        self.relation_scores_cache = {}
        self.n_templates = n_templates
        self.results_dir = results_dir
        self.dynamic_scores_output_dir = os.path.join(self.results_dir, "dynamic_scores")
        if not os.path.exists(self.dynamic_scores_output_dir):
            os.makedirs(self.dynamic_scores_output_dir)
        self.correlations_output_dir = os.path.join(self.results_dir, "correlations")
        if not os.path.exists(self.correlations_output_dir):
            os.makedirs(self.correlations_output_dir)

    def calc_dynamic_relation_scores_multiple_templates(self, relation_name, dataset):
        k = get_k(self.model_name, relation_name)
        dynamic_scores_dic_multiple_prompts = {}
        for dic in [dynamic_scores_dic_multiple_prompts]:
            for key in ["w_context_dynamic_relation_scores", 
                        "w_context_suppression_dynamic_relation_scores", 
                        "wo_context_dynamic_relation_scores", 
                        "wo_context_suppression_dynamic_relation_scores"]:
                dic[key] = []
        for template in TEMPLATES:
            dynamic_scores_dic = maps.calc_dynamic_relation_scores(dataset,template,k)
            for (dynamic_scores_key, dynamic_scores_values) in dynamic_scores_dic.items():
                dynamic_scores_dic_multiple_prompts[dynamic_scores_key].append(dynamic_scores_values)
        for (dynamic_scores_key, dynamic_scores_lst) in dynamic_scores_dic_multiple_prompts.items():
            dynamic_scores_dic_multiple_prompts[dynamic_scores_key] = np.concatenate(dynamic_scores_lst)
        return dynamic_scores_dic_multiple_prompts

    def print_pvals(self, summary_df):
        max_pval = -float('inf')
        for col in summary_df.columns:
            if "pval" in col:
                max_pval = max(max_pval, summary_df[col].max())
        print(f"{self.status()} max_pval: {max_pval:.1e}", flush=True)
        pvalues_df = summary_df[["relation"]+[col for col in summary_df.columns if "pval" in col]]
        print(pvalues_df, flush=True)

    def summary_df_to_latex(self, df):
        for is_suppression in [False,True]:
            suppression_str = "supp_" if is_suppression else ""
            df_entries = []
            for _,row in df.iterrows():
                relation_name = row["relation"]
                fixed_relation_name = relation_to_fixed_name(relation_name)
                category_name = category_to_category_name[relation_to_category[relation_name]]
                relation_scores, suppression_relation_scores = self.relation_scores_cache[relation_name] 
                max_relation_score = relation_scores.max() if (not is_suppression) else suppression_relation_scores.max()
                df_entries.append({
                    "Category": category_name,
                    "Relation": fixed_relation_name,
                    "Max relation score (over heads)": f"{max_relation_score:.2f}",
                    "Correlation w/o context": f"{row[f'{suppression_str}corr_wo_context']:.2f}",
                    "Correlation w/ context": f"{row[f'{suppression_str}corr_w_context']:.2f}",
                })
            output_df = pd.DataFrame(df_entries)  
            output_df = format_output_df(output_df)
            stripped_model_name = self.model_name.split("/")[-1]
            output_path = os.path.join(self.correlations_output_dir, f"{stripped_model_name}_{suppression_str}dynamic_scores.tex")
            output_df.to_latex(output_path, index=False)
        
    def save_results(self, summary_list):
        with open(os.path.join(self.results_dir, "templates.json"), "w") as f:
            f.write(json.dumps(TEMPLATES,indent=2))
        summary_df = pd.DataFrame(summary_list)
        summary_df.to_csv(os.path.join(self.correlations_output_dir, "summary.csv"),index=False)
        self.summary_df_to_latex(summary_df)
        self.print_pvals(summary_df)

    def save_dynamic_scores(self, dynamic_scores_dic_multiple_prompts, relation_name):
        dynamic_scores_dic = {key:val.flatten() for key,val in dynamic_scores_dic_multiple_prompts.items()}
        dynamic_scores_df = pd.DataFrame(dynamic_scores_dic)
        dir = os.path.join(self.dynamic_scores_output_dir, relation_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        dynamic_scores_df.to_csv(os.path.join(dir,"scores.csv"),index=False)

    def get_static_scores(self, relation_name, dataset):
        apply_first_mlp = True if ("Llama-3.1-70B" not in self.model_name) else False
        k = get_k(self.model_name, relation_name)
        relation_scores, suppression_relation_scores = self.maps.calc_relation_scores(dataset, apply_first_mlp, k)
        self.relation_scores_cache[relation_name] = (relation_scores, suppression_relation_scores)
        return relation_scores, suppression_relation_scores

    def calc_correlations(self, relation_name, relation_scores, suppression_relation_scores, dynamic_scores_dic_multiple_prompts):
        corr_wo_context, pval_wo = \
            pearsonr(np.concatenate([relation_scores]*self.n_templates).flatten(), dynamic_scores_dic_multiple_prompts["wo_context_dynamic_relation_scores"].flatten())
        corr_w_context, pval_w = \
            pearsonr(np.concatenate([relation_scores]*self.n_templates).flatten(), dynamic_scores_dic_multiple_prompts["w_context_dynamic_relation_scores"].flatten())
        supp_corr_wo_context, supp_pval_wo_context = \
            pearsonr(np.concatenate([suppression_relation_scores]*self.n_templates).flatten(), dynamic_scores_dic_multiple_prompts["wo_context_suppression_dynamic_relation_scores"].flatten())
        supp_corr_w_context, supp_pval_w_context = \
            pearsonr(np.concatenate([suppression_relation_scores]*self.n_templates).flatten(), dynamic_scores_dic_multiple_prompts["w_context_suppression_dynamic_relation_scores"].flatten())
        results = {
                "relation": relation_name,
                "corr_wo_context": corr_wo_context,
                "corr_w_context": corr_w_context,
                "supp_corr_wo_context": supp_corr_wo_context,
                "supp_corr_w_context": supp_corr_w_context,
                "pval_wo_context": pval_wo,
                "pval_w_context": pval_w,
                "supp_pval_wo_context": supp_pval_wo_context,
                "supp_pval_w_context": supp_pval_w_context
            }
        return results

    def run_correlative_experiment_one_relation(self, relation_name, dataset):
        print(f"{self.status()} running correlative experiment for {relation_name}")
        relation_scores, suppression_relation_scores = self.get_static_scores(relation_name, dataset)
        dynamic_scores_dic_multiple_prompts = self.calc_dynamic_relation_scores_multiple_templates(relation_name, dataset)
        self.save_dynamic_scores(dynamic_scores_dic_multiple_prompts, relation_name)
        results = self.calc_correlations(relation_name, relation_scores, suppression_relation_scores, dynamic_scores_dic_multiple_prompts)
        return results

    def run_experiment(self):
        summary_list = []
        for relation_name in tqdm(os.listdir("datasets")):
            dataset = load_dataset(relation_name)
            if not self.maps.is_valid_dataset_for_model(dataset):
                continue
            print(f"{self.status()} relation: {relation_name}")
            results = self.run_correlative_experiment_one_relation(relation_name, dataset)
            summary_list.append(results)
        self.save_results(summary_list)
        
    def status(self):
        return f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Correlative experiment for {self.model_name}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Usage: python experiment1_correlative.py <model_name>")
        sys.exit(1)
    model_name = sys.argv[1]
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] running correlative experiment for model: {model_name}")
    if "meta-llama" in model_name:
        model_name_ = "".join(model_name.split("Meta-"))
    else:
        model_name_ = model_name
    model = transformer_lens.HookedTransformer.from_pretrained_no_processing(model_name_, device_map="auto")
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    cfg = model.cfg
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    maps = MAPS(model, tokenizer)
    experiment_name = "multiple_prompts"
    results_dir = f"results/{model_name}/experiment1_correlative/{experiment_name}"
    correlative_experiment = CorrelativeExperiment(maps, model_name, experiment_name, cfg, len(TEMPLATES), results_dir)
    correlative_experiment.run_experiment()
