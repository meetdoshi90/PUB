from transformers import AutoTokenizer
import transformers
import torch
import pandas as pd
from transformers import AutoTokenizer, DataCollatorWithPadding,AutoModelForCausalLM, LlamaForCausalLM
import os
from pragmatics.process_data.data_processing_task_2 import read_data, prepare_dataset

print(torch.cuda.is_available())
from tqdm import tqdm

exp_number = "1"
def ai_harness_probs(input_ids=None, logits=None, pad_token_id=None):
    probs = logits
    idx = input_ids
    tokens_idx = idx==pad_token_id
    norm_func = torch.nn.Softmax(dim=2)
    probs = norm_func(probs)
    for i in range(idx.shape[1]):
        idx[:,i] = i*probs.shape[2] + idx[:,i]
    selected_vals = torch.take(probs,idx)
    selected_vals[tokens_idx] = 1
    mult_probs = torch.prod(selected_vals,axis=1)
    arg_max = torch.argmax(mult_probs)
    return arg_max
    

def eval():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Define the paths for data nd prompts files
    circa_data_path = '~/iclr/pragmatics/data/circa-data.tsv'
    prompt_templates_path = '~/iclr/pragmatics/prompt_templates/task_2.csv'

    # Use the functions from data_processing_task_2.py
    df, prompt = read_data(circa_data_path, prompt_templates_path)
    dataset = prepare_dataset(df, prompt)
    # print(dataset[0])
    print(dataset[0]['wrapped'])
    #print((dataset['wrapped']))

    model_path = "/raid/nlp/pranavg/iclr/pragmatics/models/llama-2-7b-chat-hf/"
    print('Loading Model...')
    #model = AutoModelForCausalLM.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    model.half()
    model.to(device)
    print('Loaded')
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    #model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    #tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    #print(tokenizer.pad_token_id,tokenizer.eos_token_id)
    #tokenizer.padding_side = "left"
    #tokenizer.truncation_side = "left"
    #model.eval()
    #model.to(device)
    options = ['A: Direct Answer', 'B: Indirect Answer']
    batch_size = 1
    num_examples = len(dataset)
    num_batches = (num_examples + batch_size - 1) // batch_size
    predictions = []
    corr_count = 0
    print('Num examples ',num_examples)
    

    # #Pipeline generate
    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     torch_dtype=torch.float16,
    #     #device_map="auto",
    #     device=0,
    #     batch_size=4,
    # )

    # sequences = pipeline(
    #     dataset['wrapped'],
    #     do_sample=False,
    #     num_return_sequences=1,
    #     eos_token_id=tokenizer.eos_token_id,
    #     max_new_tokens=1,
    #     return_full_text=False
    # )
    # pred_corr = 0
    # i = 0
    # for seq in sequences:
    #     text = seq[0]['generated_text'].strip().lower()
    #     print(text,dataset['type'][i])
    #     if dataset['type'][i]=='Direct' and text=='a':
    #         pred_corr+=1
    #     elif dataset['type'][i]=='Indirect' and text=='b':
    #         pred_corr+=1
        
    # print(pred_corr,len(dataset))



    for batch_idx in tqdm(range(num_batches)):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_examples)
        batch = dataset[batch_start:batch_end]
        input_texts = [f"{example} {option}" for example in batch['wrapped'] for option in options]
        for i in range(batch_size):
            inp = input_texts[(i*2):((i+1)*2)]
            prompt_input = tokenizer(batch['wrapped'], return_tensors='pt')
            inputs = tokenizer(inp, return_tensors="pt", padding=True, truncation=True).to(device)
            logits = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask).logits
            #logits = model.forward(**inputs).logits
            input_ids = inputs.input_ids[:,prompt_input.input_ids.shape[1]:]
            logits = logits[:,prompt_input.input_ids.shape[1]-1:-1,:]
            arg_max = ai_harness_probs(input_ids=input_ids,logits=logits,pad_token_id=tokenizer.pad_token_id)
            #print(batch['type'])
            pred = ""
            if arg_max==0:
                pred = 'Direct'
            elif arg_max==1:
                pred = 'Indirect'
            elif arg_max==2:
                pred = 'Neither'
            else:
                print('Error', arg_max)
            predictions.append(pred)
            if pred==batch['type'][0]:
                corr_count+=1
            if batch_idx%100==0:
                print(corr_count,batch_idx+1)
                print(corr_count/(batch_idx+1))

    output_dir = "/raid/nlp/pranavg/iclr/pragmatics/results" 
    output_file_name = (model_path.strip('/').split('/')[-1]).replace('-','_')
    df = dataset.to_pandas()
    df['predictions'] = predictions
    df.drop('wrapped',axis=1,inplace=True)
    df.drop('__index_level_0__',axis=1,inplace=True)
    print(df.columns)
    df.to_csv(os.path.join(output_dir,output_file_name+"_"+exp_number+".csv"), index=True)
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    precision = precision_score(df['type'], df['predictions'], average='macro')
    recall = recall_score(df['type'], df['predictions'], average='macro')
    f1_macro = f1_score(df['type'], df['predictions'], average='macro')
    accuracy = accuracy_score(df['type'], df['predictions'])

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Macro F1: {f1_macro}")
    print(f"Accuracy: {accuracy}")




    
    

            



