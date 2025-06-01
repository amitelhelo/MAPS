import os
import sys
import torch
from transformers import AutoTokenizer
import numpy as np
import transformer_lens
torch.set_default_device("cuda")
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from src.utils import get_k, load_dataset, get_topm_relation_heads
from src.paper_utils import relation_to_fixed_name, arrow
from maps import MAPS
import sys
import json
from datetime import datetime

SETUPS = [
        {"relation_name": "country_to_capital_wikidata",
        "template": "The capital of <X> is"
        },
        {"relation_name": "country_to_official_language_wikidata",
        "template": "The official language of <X> is"
        },
        {"relation_name": "product_by_company",
        "template": "Nesquik is made by Nestlé; Mustang is made by Ford; <X> is made by"
        },
        {"relation_name": "object_superclass",
        "template": "A <X> is a kind of"
        },
        {"relation_name": "adj_comparative",
        "template": " lovely-> lovelier; edgy-> edgier; <X>->"
        },
        {"relation_name": "verb_past_tense",
        "template": " hike->hiked; purchase-> purchased; <X>->"
        },
        {"relation_name": "word_first_letter",
        "template": " word-> w, o, r, d; cat-> c, a, t; <X>->"
        },
        {"relation_name": "word_last_letter",
        "template": " word-> d, r, o, w; cat-> t, a, c; <X>->"
        },
        {"relation_name": "year_to_following",
        "template": " 1300-> 1301; 1000-> 1001; <X>->"
        },
        {"relation_name": "general_copying_english_500",
        "template": " walk-> walk; cat-> cat; water-> water; <X>->"
        },
        {"relation_name": "name_copying",
        "template": " John-> John; Donna-> Donna; <X>->"
        },
        {"relation_name": "entity_to_pronoun",
        "template": " mother-> she; father-> he; tribe-> they; actress-> she; apartment-> it; <X>->"
        },
        {"relation_name": "english_to_spanish",
        "template": " apartment-> departamento; computer-> computadora; tribe-> tribu; <X>->"
        },
    ]

RELATION_NAME_TO_SETUP = {setup["relation_name"]: setup for setup in SETUPS}

class CausalExpriment:

    def __init__(self, 
                 cfg, 
                 model_name, 
                 maps: MAPS, 
                 experiment_name, 
                 n_random_sessions=5, 
                 max_control_setups=5, 
                 max_head_intersection=0.15, 
                 max_acc_diff=0.2):
        self.cfg = cfg
        self.model_name = model_name
        self.maps = maps
        self.max_head_intersection = max_head_intersection
        self.max_acc_diff = max_acc_diff
        self.n_random_sessions = n_random_sessions
        self.max_control_setups = max_control_setups
        self.max_topm_heads_to_ablate = self.get_max_topm_heads()
        self.apply_first_mlp = True if ("Llama-3.1-70B" not in self.model_name) else False
        valid_relations_for_model = self.get_valid_relations_for_model()
        self.valid_setups = [setup for setup in SETUPS if setup["relation_name"] in valid_relations_for_model]
        self.valid_relations = [setup["relation_name"] for setup in self.valid_setups]
        self.experiment_name = experiment_name
        self.results_dir = f"results/{self.model_name}/experiment2_causal/{self.experiment_name}"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def get_max_topm_heads(self):
        total_n_heads = self.cfg.n_heads * self.cfg.n_layers
        if total_n_heads < 300:
            return 30
        if total_n_heads < 500:
            return 50
        if total_n_heads < 1000:
            return 100
        if total_n_heads < 1500:
            return 150
        return 250

    def is_valid_control_task(self, setup, control_setup):
        apply_first_mlp = False if ("Llama-3.1-70B" in self.model_name) else True
        relation_name = setup["relation_name"]
        dataset = load_dataset(relation_name)
        template = setup["template"]
        control_template = control_setup["template"]
        control_relation_name = control_setup["relation_name"]
        control_dataset = load_dataset(control_relation_name)
        topm_relation_heads = get_topm_relation_heads(self.max_topm_heads_to_ablate, self.maps, dataset, apply_first_mlp, get_k(self.model_name, relation_name), only_nonzero=True)
        control_topm_relation_heads = get_topm_relation_heads(len(topm_relation_heads), self.maps, control_dataset, apply_first_mlp, get_k(self.model_name, control_relation_name), only_nonzero=True)
        topm_relation_heads = set(topm_relation_heads)
        control_topm_relation_heads = set(control_topm_relation_heads)
        heads_intersection = len(topm_relation_heads.intersection(control_topm_relation_heads)) / len(topm_relation_heads)
        heads_intersection_term =  heads_intersection < self.max_head_intersection
        main_accuracy = self.maps.calc_causal_effects(dataset, template, [])[0]
        control_accuracy = self.maps.calc_causal_effects(control_dataset, control_template, [])[0]
        accuracy_term = abs(main_accuracy - control_accuracy) < self.max_acc_diff
        validity = heads_intersection_term and accuracy_term
        return main_accuracy, control_accuracy, heads_intersection, validity

    def sample_m_random_heads(self, m, heads_to_exclude):
        heads_to_exclude_indexed = [layer*self.cfg.n_heads + head for (layer,head) in heads_to_exclude]
        all_heads = [ix for ix in range(self.cfg.n_layers*self.cfg.n_heads)]
        available_heads = list(set(all_heads) - set(heads_to_exclude_indexed))
        sampled_heads_indexed = np.random.choice(available_heads, size=m, replace=False)
        sampled_heads = [(ix // self.cfg.n_heads, ix % self.cfg.n_heads) for ix in sampled_heads_indexed]
        return sampled_heads

    def get_control_setups(self, relation_name):
        setup = RELATION_NAME_TO_SETUP[relation_name]
        optional_control_setups = self.valid_setups.copy()
        np.random.shuffle(optional_control_setups)
        control_setups = []
        for optional_control_setup in optional_control_setups:
            if relation_name == optional_control_setup["relation_name"]:
                continue
            if self.is_valid_control_task(setup, optional_control_setup)[-1]:
                control_setups.append(optional_control_setup)
                if len(control_setups) == self.max_control_setups:
                    break
        return control_setups

    def prepare_accuracies_df(self, lsts_of_accuracies, heads_removed):
        df_entries = []
        if len(lsts_of_accuracies) == 0:
            return pd.DataFrame({"k_heads_ablated":[],"accuracy":[]})
        n_heads = len(lsts_of_accuracies[0])
        for k_heads_ablated in range(n_heads):
            for lst_ix,lst_of_accuracies in enumerate(lsts_of_accuracies):
                heads_removed_lst = [""] + heads_removed[lst_ix]
                df_entries.append({
                        "k_heads_ablated": k_heads_ablated,
                        "accuracy": lst_of_accuracies[k_heads_ablated],
                        "last_head_removed": str(heads_removed_lst[k_heads_ablated])
                    })
        return pd.DataFrame(df_entries)
    
    def get_valid_relations_for_model(self):
        valid_relations_for_model = []
        for relation_name in os.listdir("datasets"):
            dataset = load_dataset(relation_name)
            if self.maps.is_valid_dataset_for_model(dataset):
                valid_relations_for_model.append(relation_name)
        return valid_relations_for_model

    def save_relation_results(self, 
                     relation_name,
                     relation_heads_accuracies_df, 
                     random_heads_accuracies_df, 
                     control_heads_accuracies_df,
                     control_setups):
        sns.lineplot(relation_heads_accuracies_df, x="k_heads_ablated", y="accuracy", label="ablating relation heads")
        sns.lineplot(random_heads_accuracies_df, x="k_heads_ablated", y="accuracy", label="ablating random heads")
        sns.lineplot(control_heads_accuracies_df, x="k_heads_ablated", y="accuracy", label="ablating relation heads, control tasks")
        setup = RELATION_NAME_TO_SETUP[relation_name]
        plt.title(f"MAPS causal experiment\nrelation={relation_name}\ntemplate={setup['template']}\n{model_name}")
        plt.ylabel("Accuracy")
        plt.yticks(np.arange(0,1.05,0.1))
        plt.xlabel("# heads ablated")
        relation_dir_path = os.path.join(self.results_dir, "relations_results", relation_name)
        if not os.path.exists(relation_dir_path):
            os.makedirs(relation_dir_path)
        plt.savefig(os.path.join(relation_dir_path,f"{relation_name}.png"), bbox_inches='tight')
        plt.close()
        relation_heads_accuracies_df.to_csv(os.path.join(relation_dir_path, "relation_heads_accuracies_df.csv"), index=False)
        random_heads_accuracies_df.to_csv(os.path.join(relation_dir_path, "random_heads_accuracies_df.csv"), index=False)
        control_heads_accuracies_df.to_csv(os.path.join(relation_dir_path, "control_heads_accuracies_df.csv"), index=False)
        with open(os.path.join(relation_dir_path, "control_setups.json"), "w") as f:
            f.write(json.dumps(control_setups,indent=2))

    def prepare_latex_one_relation(self, 
                               relation_name,
                               control_baselines_accuracies,
                               important_heads_accuracies, 
                               random_baselines_accuracies):
        n_heads_ablated = important_heads_accuracies.shape[0] - 1
        heads_with_zero_acc = np.where(important_heads_accuracies["accuracy"]==0)[0]
        if len(heads_with_zero_acc > 0):
            n_heads_ablated = heads_with_zero_acc[0]
        first_acc = important_heads_accuracies[important_heads_accuracies["k_heads_ablated"]==0]['accuracy'].item()
        if first_acc < 0.1:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] skipping relation: {relation_name}, because initial acc is small", flush=True)
            return None
        final_acc = important_heads_accuracies[important_heads_accuracies["k_heads_ablated"]==n_heads_ablated]['accuracy'].item()
        final_random_accuracies = random_baselines_accuracies[random_baselines_accuracies["k_heads_ablated"]==n_heads_ablated]["accuracy"]
        final_random_accuracies = np.array(final_random_accuracies)
        final_random_mean, final_random_std = np.mean(final_random_accuracies), np.std(final_random_accuracies)
        if control_baselines_accuracies.shape[0] > 0:
            control_first_accuracies = np.array(control_baselines_accuracies[control_baselines_accuracies["k_heads_ablated"]==0]["accuracy"])
            control_first_acc_mean, control_first_acc_std = np.mean(control_first_accuracies), np.std(control_first_accuracies)
            final_control_accs = np.array(control_baselines_accuracies[control_baselines_accuracies["k_heads_ablated"]==n_heads_ablated]["accuracy"])
            control_final_acc_mean, control_final_acc_std = np.mean(final_control_accs), np.std(final_control_accs)
            control_acc_change = ((control_first_acc_mean - control_final_acc_mean) / control_first_acc_mean)*100
        acc_change = ((first_acc-final_acc)/first_acc)*100
        random_acc_change = ((first_acc-final_random_mean)/first_acc)*100
        entry = {
            r"\makecell{Relation name}": relation_to_fixed_name(relation_name),
            r"\makecell{\#\\Heads\\removed}":  n_heads_ablated,
            r"\makecell{Base}": f"{first_acc:.2f}",
            r"\makecell{-TR}": f"\\tcbox{{{arrow(acc_change)}{abs(acc_change):.0f}\\%}}{final_acc:.2f}",
            r"\makecell{-RND}": f"\\tcbox{{{arrow(random_acc_change)}{abs(random_acc_change):.0f}\\%}}{final_random_mean:.2f} $\\pm$ {final_random_std:.2f}",
            r"\makecell{\# tasks}": sum(control_baselines_accuracies["k_heads_ablated"]==0),
            r"\makecell{Base (CTR)}": None,
            r"\makecell{-TR (CTR)}": None,
        }
        if entry[r"\makecell{\# tasks}"] > 0:
            entry[r"\makecell{Base (CTR)}"] = f"{control_first_acc_mean:.2f} $\\pm$ {control_first_acc_std:.2f}"
            entry[r"\makecell{-TR (CTR)}"] = f"\\tcbox{{{arrow(control_acc_change)}{abs(control_acc_change):.0f}\\%}}{control_final_acc_mean:.2f} $\\pm$ {control_final_acc_std:.2f}"
        return entry

    # Note: results may deviate a bit from the results presented in the paper because of
    # two sources of randomness: different random heads sampled on each run, different control tasks sampled on each run
    def save_results(self, relation_to_experiment_results):
        df_entries = []
        relation_names = relation_to_experiment_results.keys()
        for relation_name in relation_names:
            control_baselines_accuracies = relation_to_experiment_results[relation_name]["control_heads_accuracies_df"]
            important_heads_accuracies = relation_to_experiment_results[relation_name]["relation_heads_accuracies_df"]
            random_baselines_accuracies = relation_to_experiment_results[relation_name]["random_heads_accuracies_df"]
            entry = self.prepare_latex_one_relation(relation_name,
                                    control_baselines_accuracies,
                                    important_heads_accuracies,
                                    random_baselines_accuracies)
            if entry:
                df_entries.append(entry)
        df = pd.DataFrame(df_entries)
        df = df.sort_values(by=r"\makecell{Relation name}")
        df = df.fillna("")
        df.to_csv(os.path.join(self.results_dir, "summary_df.csv"))
        latex = df.to_latex(
            os.path.join(self.results_dir, "summary_df.tex"),
            column_format="lrrrrrrr",
            index=False, 
            float_format=lambda x: f"{x:.1f}"
            )
        with open(os.path.join(self.results_dir, "setups.json"), "w") as f:
            f.write(json.dumps(self.valid_setups, indent=2))
        return df, latex

    def run_experiment_one_relation(self, relation_name):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Running causal experiment - {self.model_name} - {relation_name} - ablating relation heads", flush=True)
        setup = RELATION_NAME_TO_SETUP[relation_name]
        control_setups = self.get_control_setups(relation_name)
        dataset = load_dataset(relation_name)
        relation_heads = get_topm_relation_heads(self.max_topm_heads_to_ablate, self.maps, dataset, self.apply_first_mlp, get_k(self.model_name, relation_name), only_nonzero=True)
        relation_heads_accuracies = [self.maps.calc_causal_effects(dataset, setup["template"], relation_heads)]
        control_heads_accuracies = []
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Running causal experiment - {self.model_name} - {relation_name} - ablating control heads", flush=True)
        for control_setup in control_setups:
            control_dataset = load_dataset(control_setup["relation_name"])
            control_heads_accuracies.append(self.maps.calc_causal_effects(control_dataset, control_setup["template"], relation_heads))
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Running causal experiment - {self.model_name} - {relation_name} - ablating random heads", flush=True)
        random_heads_accuracies = []
        random_heads_lsts = []
        for _ in range(self.n_random_sessions):
            random_heads = self.sample_m_random_heads(len(relation_heads), relation_heads)
            random_heads_accuracies.append(self.maps.calc_causal_effects(dataset, setup["template"], random_heads))
            random_heads_lsts.append(random_heads)
        relation_heads_accuracies_df = self.prepare_accuracies_df(relation_heads_accuracies, [relation_heads])
        control_heads_accuracies_df = self.prepare_accuracies_df(control_heads_accuracies, [relation_heads]*len(control_setups))
        random_heads_accuracies_df = self.prepare_accuracies_df(random_heads_accuracies, random_heads_lsts)
        return relation_heads_accuracies_df, random_heads_accuracies_df, control_heads_accuracies_df, control_setups

    def run_experiment(self):
        pbar = tqdm(self.valid_relations)
        relation_to_experiment_results = {}
        for relation_name in pbar:
            pbar.set_description(f"Running causal experiment for: {relation_name}")
            relation_heads_accuracies_df, random_heads_accuracies_df, control_heads_accuracies_df, control_setups = \
                            causal_experiment.run_experiment_one_relation(relation_name)
            relation_to_experiment_results[relation_name] = {
                "relation_heads_accuracies_df": relation_heads_accuracies_df,
                "random_heads_accuracies_df": random_heads_accuracies_df,
                "control_heads_accuracies_df": control_heads_accuracies_df
            }
            self.save_relation_results(relation_name, relation_heads_accuracies_df, random_heads_accuracies_df, control_heads_accuracies_df, control_setups)
        self.save_results(relation_to_experiment_results)
        
        

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Usage: python3 experiment2_causal.py <model_name>", flush=True)
        sys.exit(1)
    model_name = sys.argv[1]
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Started causal experiment for {model_name}", flush=True)
    experiment_name = "test"
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
    causal_experiment = CausalExpriment(cfg, model_name, maps, experiment_name)
    causal_experiment.run_experiment()
    