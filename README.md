## MAPS
Code for "Inferring Functionality of Attention Heads from their Parameters" - Amit Elhelo, Mor Geva. 2024 

## USAGE
The demo and salient_operations notebooks are designed to be straightforward and easy to explore.

Each of the main experiments: experiment1_correlative, experiment2_causal, and multi_token accepts a single argument: the model identifier from Hugging Face (the model needs to also be implemented by transformer-lens).
Example: python experiment1_correlative.py EleutherAI/pythia-6.9b

## LICENSES
This project uses transformer-lens, licensed under the MIT License. It also relies on Hugging Face transformers, licensed under the Apache 2.0 License.
Each dataset used in this project includes its own README file detailing: origin, license information, main modifications made
