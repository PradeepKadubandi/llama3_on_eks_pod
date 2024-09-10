import torch
import os
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import datasets
from pathlib import Path

save_dir="/shared/"
model = LlamaForCausalLM.from_pretrained("NousResearch/Meta-Llama-3.1-8B")
torch.save(model.state_dict(), os.path.combine(save_dir, "llama3.1-8b-hf-pretrained.pt"))

#downloading dolly dataset
dataset="databricks/databricks-dolly-15k" 
target_dataset_path=save_dir+dataset
datasets.config.DOWNLOADED_DATASETS_PATH = Path(target_dataset_path)
datasets.load_dataset(dataset)
