import os
import re
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    pipeline
    
)
import json
from tqdm import tqdm
import traceback

from src.model_5W1H.evaluation_metric_template_kaggle_subtask_1 import eval_dataset


with open('./data/5W1H/test_split_BIO_inline.json') as f:
    dataset = json.load(f)

systemPrompt = dataset['system']
dataset = pd.DataFrame(dataset['prompts'])

#dataset = dataset[0:20]

print(systemPrompt)
#print(dataset)

messages = [
    [
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": instruction["user"]},
    ]
    for i, instruction in dataset.iterrows()
]


#PARAMS
MODELS_DIR = './models/5W1H/Salamandra_2B/5_epoch/hf_models/'

dataGold = pd.read_csv("./data/5W1H/5w1h_subtarea_1_submission_downloaded.csv")
dataText = pd.read_json("./data/5W1H/5w1h_subtarea_1_test.json", lines = True)
print(dataText)

# Elimina la parte del texto que corresponde al prompt 
def clean_output_LLAMA(text):
    return text.split("assistant")[1]

# Extrae las etiquetas y sus valores
def extract_tags_and_values(text):
    tags = re.findall(r'\[(.*?)\](.*?)\[\/\1\]', text)
    result = {}
    
    for tag, value in tags:
        if tag not in result:
            result[tag] = []
        result[tag].append(value.strip())
    return result


tokenizer = AutoTokenizer.from_pretrained(
        "BSC-LT/salamandra-2b-instruct",
        #token=os.getenv("HUGGINGFACE_APIKEY"),
        use_fast=False,
        padding_side="right",
    )


#Applying different LoRA weights to the base model, these weights were accomplished with Fine-Tuning 
for model in os.listdir(MODELS_DIR):
    model_dir = os.path.join(MODELS_DIR, model)
    if os.path.isdir(model_dir):
        print("MODEL:", model_dir)
            #-------------------INFERENCE----------------------
        pipe = pipeline(
            "text-generation",
            model=model_dir,
            tokenizer=tokenizer,
            #token=os.getenv("HUGGINGFACE_APIKEY"),
            max_length=512,
            device_map="auto",
        )
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

        model_outputs = []
        generate_kwargs = {
            "do_sample": False,
            "temperature": 1e-3,
            "top_p": 0.9,
            "max_new_tokens": 512,
        }

        

        #Generating results from checkpoint

        batch_size = 8
        for i in tqdm(range(0, len(messages), batch_size)):
            batch_data = messages[i : i + batch_size]
            out = pipe(
                batch_data,
                batch_size=batch_size,
                truncation="only_first",
                pad_token_id=pipe.tokenizer.eos_token_id,
                **generate_kwargs,
            )
            model_outputs.extend(out)

        assistant_outputs = [out[0]['generated_text'][2]["content"] for out in model_outputs]
        #out[0]["generated_text"][2]["content"]
        #out["generated_text"][0]['assistant']
    
        #---------------------END_INFERENCE-------------------------------
        result_df = pd.DataFrame(columns =['clean_output'], data = assistant_outputs)
        result_df['id'] = dataset['Id'].values
        result_df['text'] = dataText['Text'].values
        
        result_df.to_csv(f'{MODELS_DIR}/{model}.csv', index = False) #Guardamos la salida
        
        #Process the DataFrame to convert it to a JSON format
        df_eval = result_df
        #df_eval['clean_output'] = result_df['output'].apply(clean_output_LLAMA) #Limpiando salida
        print(df_eval)
        new_rows = []

        for idx, row in df_eval.iterrows():
            id_value = row['id']
            text_value = row['text']
            clean_output_value = row['clean_output']
            
            extracted_data = extract_tags_and_values(clean_output_value) #Extrae las etiquetas
            
            for tag, values in extracted_data.items():
                new_rows.append({
                    'id': id_value,
                    'text': text_value,
                    'tag': tag,
                    'values': values
                })
        df_test = pd.DataFrame(new_rows)

        df = df_test
        # df['values'] = df['values'].apply(lambda x:  literal_eval(x))
        df['id'] = df['id'].astype(int)

        output = []

        for (id_, text), group in df.groupby(['id', 'text']):
            tags = []
            
            for index, row in group.iterrows():
                for value in row['values']:

                    # Encuentra el inicio y fin de cada etiqueta en el texto
                    tag_start = text.find(value)
                    if tag_start < 0:
                        tag_start = 0
                    tag_end = tag_start + len(value)
                    if row['tag'] in ['WHAT', 'WHEN', 'WHERE', 'WHO', 'WHY', 'HOW']:
                        tag_dict = {
                            "Tag_Start": int(tag_start), #Formato del conjunto de evaluación
                            "Tag_End": int(tag_end), #Formato del conjunto de evaluación
                            "5W1H_Label": str(row['tag']), #Etiqueta asignada 5W1H
                            "Reliability_Label": "confiable", # inventada
                            "Tag_Text": str(value)
                        }
                        tags.append(tag_dict)

            output.append({
                "Id": int(id_),
                "Text": str(text),
                "Tags": tags
            })
        with open(f'{MODELS_DIR}/test_{model}.json', 'w') as file:
            for dictionary in output:
                json.dump(dictionary, file)
                file.write('\n')

        df_output = pd.DataFrame(output)
        df_output.to_csv(f'{MODELS_DIR}/test_{model}.csv', sep=',', index = False)
        
        #-------------------COMPARING RESULTS WITH GOLD_LABEL----------------------
        # try:
        #     output = pd.read_json(f'{MODELS_DIR}/{model}.json', lines = True)
        #     output['Tags'] = output['Tags'].astype(str)
        #     output = output[["Id","Tags"]]
            
        #     df, metrics = eval_dataset(dataGold, output, ['WHAT', 'WHEN', 'WHERE', 'WHO', 'WHY', 'HOW'])

        #     with open(f'{MODELS_DIR}/{model}.txt', 'w') as file:
        #         file.write(str(metrics))
        # except Exception as e:
        #     print("Error:", e)
        #     traceback.print_exc()
