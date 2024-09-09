# import torch
# import os
# from transformers.models.llama.modeling_llama import LlamaForCausalLM
# import os

# save_dir = "/shared/"
# #model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
# model = LlamaForCausalLM.from_pretrained("NousResearch/Meta-Llama-3-8B")
# torch.save(model.state_dict(), save_dir + "llama3-8b-hf-pretrained.pt")

import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import datasets
from pathlib import Path

#downloading llama3 model
save_dir="/shared/"
model = LlamaForCausalLM.from_pretrained("NousResearch/Meta-Llama-3-8B")
torch.save(model.state_dict(), save_dir + "llama3-8b-hf-pretrained.pt")

#downloading dolly dataset
dataset="databricks/databricks-dolly-15k" 
target_dataset_path=save_dir+dataset
datasets.config.DOWNLOADED_DATASETS_PATH = Path(target_dataset_path)
datasets.load_dataset(dataset)
