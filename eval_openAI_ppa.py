import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
import csv
import openai
from tqdm.auto import tqdm
from pragmatics.eval_tasks.prompt_script import get_data
from datasets import Dataset
import ast
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time
import json

openai.api_key = ""


def evaluate(dataset,metadata):
    k = 0
    data_len = len(dataset)
    outputs = []
    file_name = "output_zero.json"
    print(data_len)
    i=0
    num_options = len(dataset['answer_options'][0])
    with tqdm(total=data_len) as pbar:
        while i<data_len:

            try:
                response = openai.Completion.create(
                    model="gpt-3.5-turbo-instruct",
                    #messages=[
                    #    {"role": "system", "content": "You are a helpful assistant."},
                    #    {"role": "user", "content": dataset['wrapped'][i]}    
                    #],
                    prompt = dataset['wrapped'][i],
                    max_tokens=1,
                    temperature=1.0,
                    logprobs=5,
                )
            except Exception as e:
                if e: 
                    print(e)   
                    print('Timeout error, retrying...')
                    time.sleep(180)    
                else:    
                    raise e 
            if response and response["choices"] and response["choices"][0] and response["choices"][0]["logprobs"] and response["choices"][0]["logprobs"]["top_logprobs"] :
                result_dict = {}
                logprobs = dict(response["choices"][0]["logprobs"]["top_logprobs"][0])
                options = ast.literal_eval(dataset['options'][i])
                rrd = {'prompt':dataset['wrapped'][i],'probs':logprobs,'correct_answer':chr(options.index(dataset['correct answer'][i])+ord('A'))}
                with open(file_name, 'a') as file:
                    json.dump(rrd, file)
                    file.write('\n')
                for key,value in logprobs.items():
                    cleaned_key = key.strip().upper()
                    if cleaned_key.isalpha():
                        if cleaned_key in result_dict:
                            result_dict[cleaned_key] = max([result_dict[cleaned_key],value])
                        else:
                            result_dict[cleaned_key] = value
                
                outputs.append(max(result_dict, key=result_dict.get))
                i+=1
                pbar.update(1)
            else:
                k+=1
                generated_response = "ERROR"
                print(generated_response)
                
    print(k)
    df = dataset.to_pandas()
    options = [ast.literal_eval(o) for o in df['options']]
    correct_options = [chr(options[i].index(df['correct answer'][i])+ord('A')) for i in range(len(df['correct answer']))]
    df['correct_options'] = correct_options
    predictions = [outputs[i][0] for i in range(len(outputs))]
    df['generate_predictions'] = predictions
    accuracy = accuracy_score(df['correct_options'], df['generate_predictions'])
    precision = precision_score(df['correct_options'], df['generate_predictions'],average='macro')
    recall = recall_score(df['correct_options'], df['generate_predictions'],average='macro')
    f1_macro = f1_score(df['correct_options'], df['generate_predictions'],average='macro')
    results_dict = {
    'precision':precision,
    'recall':recall,
    'f1_macro':f1_macro,
    'accuracy':accuracy
    }
    
    if not os.path.exists("gpt3_results_zero_shot"):
        os.makedirs("gpt3_results_zero_shot")
    absolute_metric_path = os.path.join("gpt3_results_zero_shot",str(metadata['task']) + ".csv")
    absolute_result_path = os.path.join("gpt3_results_zero_shot","data_" +str(metadata['task']) + ".csv")
    df.to_csv(absolute_result_path, index=True)
    with open(absolute_metric_path, 'w') as json_file:
        json.dump(results_dict, json_file)

if __name__=="__main__":
    tasks = [0,1,2,3,4,5,6,7,8,9,10,13,14,3,11,15]
    # tasks = [4]
    metadata= {
        'task':0,
        'model':'gpt2',
        'input_dir':'./pragmatics/global_datasets/',
        'model_path':'/raid/nlp/pranavg/iclr/pragmatics/models/llama-2-13b-chat-hf/',
        'prompt':'zero_shot', # zero_shot to few_shot
        'run':0,
        'output_dir':'/raid/nlp/pranavg/iclr/Results',
        'eval_method':'mcqa', #mcqa or harness or ppa
        'k':0 # 0,3,5
    }
    for t in tasks:
        print(metadata['model'])
        print(metadata['prompt'])
        print(metadata['k'])
        print(metadata['eval_method'])
        print(t)

        metadata['task'] = t
        data_file_name = f"task_{metadata['task']}.csv"
        data_path = os.path.join(metadata['input_dir'],data_file_name)
        gdata = pd.read_csv(data_path)
        dataset = get_data(gdata,metadata)
        dataset_pd = dataset
        # dataset = dataset[:5]
        dataset = Dataset.from_pandas(dataset)
        print("-------")
        print(type(dataset['wrapped']),len(dataset))
        print("-----")
        evaluate(dataset,metadata)
