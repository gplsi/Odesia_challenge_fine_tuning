from transformers import AutoTokenizer
import os
import json
from prepare_data.preprocess import tokenizer_dataset

config_model = "../data/5W1H/mock_model_config.json"

# Load the model config
#with open(os.getcwd() + config_model) as f:
#    config_model = json.load(f)

with open(config_model) as f:
    config_model = json.load(f)

#Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(config_model["model_name"])

#Tokenize the dataset
train_dataset = tokenizer_dataset(config_model["training_data"], tokenizer, config_model, config_model['train_args']['max_seq_length'])
val_dataset   = tokenizer_dataset(config_model["eval_data"],     tokenizer, config_model, config_model['train_args']['max_seq_length'])

#Save the tokenized dataset
train_dataset.save_to_disk(config_model["training_tokenized"])
val_dataset.save_to_disk(config_model["eval_tokenized"])