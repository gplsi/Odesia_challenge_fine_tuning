import os
import pandas as pd
import re
import torch
from torch.utils.data import DataLoader
import datasets
from transformers import DataCollatorForLanguageModeling
from datasets import load_from_disk

def de_emojify(text):
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F92F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        u"\U00002702-\U000027B0"
                                        u"\U000024C2-\U0001F251"
                                        u"\U0001F190-\U0001F1FF"
                                        u"\U0001F926-\U0001FA9F"
                                        u"\u2640-\u2642"
                                        u"\u2600-\u2B55"
                                        u"\u200d"
                                        u"\u23cf"
                                        u"\u23e9"
                                        u"\u231a"
                                        u"\ufe0f"
                                        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


def preprocess(value):
    new_value = de_emojify(value)
    new_value = re.sub(r'http\S+', '', new_value)
    return new_value


def load_all_data(file_in, label, labels_to_exclude=[], filter_label=None, filter_label_value=None, is_preprocess=False):
    if file_in.endswith('.tsv'):
        df_in = pd.read_csv(os.getcwd() + file_in, sep='\t')
    else:
        df_in = pd.read_json(os.getcwd() + file_in, lines=True)

    for value in labels_to_exclude:
        df_in = df_in[df_in[label] != value]
    if label in df_in.columns:
        if filter_label:
            df_in = df_in[df_in[filter_label] == filter_label_value]
        print(df_in[label].value_counts())
        labels = df_in[label]
    if is_preprocess:
        df_in['text'] = df_in['text'].apply(preprocess)
    list_of_tuples = list(zip(list(df_in['text']), list(labels)))
    df = pd.DataFrame(list_of_tuples, columns=['text', 'label'])
    return df


def preprocess_function_conditional(sample, tokenizer, max_source_length, max_target_length, padding="max_length"):
    # add prefix to the input for t5
    inputs = [
        "Clasifica el siguiente tuit en una categoría: "
        + item.replace("\n", " ")
        + " La respuesta es: "
        for item in sample["text"]
    ]

    # tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=padding,
        truncation=True,
        return_tensors="pt",
        return_special_tokens_mask=True
    )

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=sample["label"],
        max_length=max_target_length,
        padding=padding,
        truncation=True,
        return_special_tokens_mask=True,
    )

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        labels = torch.tensor(labels, dtype=torch.long)

    model_inputs["labels"] = labels
    return model_inputs


def preprocess_function_causal(sample, tokenizer, max_source_length, max_target_length, padding="max_length"):
    inputs_label = [" La respuesta es: " + label for label in sample['label']]
    inputs = [
        "Clasifica el siguiente tuit en una categoría: "
        + text.replace("\n", " ")
        for text in sample['text']
    ]

    model_inputs = tokenizer.batch_encode_plus(list(zip(inputs, inputs_label)), max_length=max_source_length + max_target_length, padding=padding,
                          truncation_strategy='only_first', return_tensors="pt")

    model_inputs["labels"] = model_inputs["input_ids"].clone()
    return model_inputs


def get_dataloader(df, config_model, tokenizer, shuffle=False):
    # Create the dataset
    train_dataset = datasets.Dataset.from_pandas(df)
    # Preprocess the data
    preprocess_function = preprocess_function_causal if config_model["llm_type"] == "causal" else preprocess_function_conditional
    train_dataset = train_dataset.map(
        preprocess_function, batched=True, remove_columns=["text", "label"],
        fn_kwargs={"tokenizer": tokenizer, "max_source_length": config_model["max_source_length"],
                   "max_target_length": config_model["max_target_length"]})
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    # Create the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=config_model["batch_size"], shuffle=shuffle)
    #train_dataloader = train_dataset
    return train_dataloader

def save_dataset(df, config_model, tokenizer, file_name):
    # Create the dataset
    train_dataset = datasets.Dataset.from_pandas(df)
    # Preprocess the data
    preprocess_function = preprocess_function_causal if config_model["llm_type"] == "causal" else preprocess_function_conditional
    train_dataset = train_dataset.map(
        preprocess_function, batched=True, remove_columns=["text", "label"],
        fn_kwargs={"tokenizer": tokenizer, "max_source_length": config_model["max_source_length"],
                   "max_target_length": config_model["max_target_length"]})
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    train_dataset.save_to_disk(file_name)


def create_dataloader_from_disk(file_dir, config_model, shuffle=False):
    print(os.path.join(os.getcwd(), file_dir))
    train_dataset = load_from_disk(os.path.join(os.getcwd(), file_dir))

    return DataLoader(train_dataset, batch_size=config_model["batch_size"], shuffle=shuffle)


