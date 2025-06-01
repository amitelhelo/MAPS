import numpy as np
from enum import Enum
import pandas as pd
import os
from src.utils import load_dataset, get_k

class RelationCategory(Enum):
    ALGORITHMIC = 1
    KNOWLEDGE = 2
    LINGUISTIC = 3
    TRANSLATION = 4

category_to_category_name = {
    RelationCategory.ALGORITHMIC: "Algorithmic",
    RelationCategory.KNOWLEDGE: "Knowledge",
    RelationCategory.LINGUISTIC: "Linguistic",
    RelationCategory.TRANSLATION: "Translation"
}

relation_to_category = {
'general_copying_english_500': RelationCategory.ALGORITHMIC,
'year_to_following': RelationCategory.ALGORITHMIC,
'name_copying': RelationCategory.ALGORITHMIC,
'word_first_letter': RelationCategory.ALGORITHMIC,
'word_last_letter': RelationCategory.ALGORITHMIC,

 'work_location': RelationCategory.KNOWLEDGE,
 'object_superclass': RelationCategory.KNOWLEDGE,
 'product_by_company': RelationCategory.KNOWLEDGE,
 'country_to_official_language_wikidata': RelationCategory.KNOWLEDGE,
 'country_to_capital_wikidata': RelationCategory.KNOWLEDGE,

 'adj_antonym': RelationCategory.LINGUISTIC,
 'adj_comparative': RelationCategory.LINGUISTIC,
 'adj_superlative': RelationCategory.LINGUISTIC,
 'verb_past_tense': RelationCategory.LINGUISTIC,
 'entity_to_pronoun': RelationCategory.LINGUISTIC,
 'word_to_synonym': RelationCategory.LINGUISTIC,
 'word_to_homophone': RelationCategory.LINGUISTIC,
 'word_to_compound': RelationCategory.LINGUISTIC,

 'english_to_spanish': RelationCategory.TRANSLATION,
 'english_to_french': RelationCategory.TRANSLATION,
}

def format_output_df(output_df):
    output_df = output_df[[
        'Category',
        'Relation',
        'Correlation w/o context', 
        'Correlation w/ context',
        'Max relation score (over heads)',
        ]]
    output_df.columns = [
        'Category',
        'Relation',
        r'\makecell{Correlation\\w/o context}', 
        r'\makecell{Correlation\\w/ context}',
        r'\makecell{Max relation score\\(over heads)}',
        ]
    output_df = output_df.sort_values(by=["Category", "Relation"])
    already_formatted = {
    "Algorithmic":False,
    "Knowledge": False,
    "Linguistic": False,
    "Translation": False
    }
    category_lengths = output_df['Category'].value_counts().to_dict()
    def format_category(category):
        if already_formatted[category]:
            return ""
        else:
            already_formatted[category] = True
            category_length = category_lengths[category]
            return r"\multirow{category_length}{*}{category}".replace("category_length", str(category_length)).replace("category",category)
    output_df['Category'] = output_df['Category'].apply(format_category)
    return output_df

def model_name_to_printing_name(model_name):
    if model_name == "EleutherAI/pythia-6.9b":
        return "Pythia 6.9B"
    if model_name == "EleutherAI/pythia-12b":
        return "Pythia 12B"
    if model_name == "microsoft/phi-2":
        return "Phi-2"
    if model_name == "gpt2-xl":
        return "GPT-2 xl"
    if model_name == "gpt2":
        return "GPT-2"
    if model_name == "gpt2-medium":
        return "GPT-2 medium"
    if model_name == "gpt2-large":
        return "GPT-2 large"
    if model_name == "meta-llama/Meta-Llama-3.1-70B":
        return "Llama-3.1 70B"
    if model_name == "meta-llama/Meta-Llama-3.1-8B":
        return "Llama-3.1 8B"
    if model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        return "Llama-3.1 8B (Instruct)"
    raise Exception(f"invalid model name {model_name}")

def relation_to_fixed_name(relation_name):
    if "ANTI_" in relation_name:
        return "(suppression) " + relation_to_fixed_name(relation_name.split("ANTI_")[-1])
    if relation_name == "adj_antonym":
        return "Word to antonym"
    if relation_name == "adj_comparative":
        return "Adj to comparative"
    if relation_name == "adj_superlative":
        return "Adj to superlative"
    if relation_name == "country_to_capital_wikidata":
        return "Country to capital"
    if relation_name == "country_to_official_language_wikidata":
        return "Country to language"
    if relation_name == "english_to_french":
        return "English to French"
    if relation_name == "english_to_spanish":
        return "English to Spanish"
    if relation_name == "entity_to_pronoun":
        return "Noun to pronoun"
    if relation_name == "general_copying_english_500":
        return "Copying"
    if relation_name == "name_copying":
        return "Name copying"
    if relation_name == "object_superclass":
        return "Object to superclass"
    if relation_name == "product_by_company":
        return "Product by company"
    if relation_name == "verb_past_tense":
        return "Verb to past tense"
    if relation_name == "word_to_compound":
        return "Word to compound"
    if relation_name == "word_to_homophone":
        return "Word to homophone"
    if relation_name == "word_to_synonym":
        return "Word to synonym"
    if relation_name == "work_location":
        return "Work to location"
    if relation_name == "year_to_following":
        return "Year to following"
    if relation_name == "word_first_letter":
        return "Word to first letter"
    if relation_name == "word_last_letter":
        return "Word to last letter"


def arrow(val):
    return r'$\downarrow$' if val >= 0 else r'$\uparrow$'

