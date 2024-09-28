import torch
from tqdm import tqdm
import os
import ast
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
import math
import itertools
import numpy as np
import torch

def ai_harness_probs(input_ids=None, logits=None, pad_token_id=None, normalisation='length', unconditional_input_ids=None, unconditional_logits=None):
    # Normalize using Softmax
    #print('Logits shape',logits.shape)
    norm_func = torch.nn.Softmax(dim=2)
    probs = norm_func(logits) # B * C * V
    
    # Find indices corresponding to padding token
    idx = input_ids # B * C
    tokens_idx = idx==pad_token_id
    #print(probs)

    rev_token_mask = ~tokens_idx
    norm_value = torch.sum(rev_token_mask,axis=1) #Get length

    # Select the values based on indices and set the probability to 1
    for i in range(idx.shape[0]):
        for j in range(idx.shape[1]):
            idx[i,j] = i*probs.shape[1]*probs.shape[2] + j*probs.shape[2] + idx[i,j] #Creating indexes for torch.take
    
    selected_vals = torch.take(probs,idx)
    selected_vals[tokens_idx] = 1.0
    
    # Multiply and get the argument with maximum product
    #print(selected_vals.shape,selected_vals)
    mult_probs = torch.prod(selected_vals,axis=1)
    
    #Normalisation
    if normalisation == 'length': # Length normalise
        normalised_probs = torch.pow(mult_probs,1/norm_value) #Taking nth root of the length of the answer
    else: # Unconditional prob normalise
        # Same process repeat for second pass to calculate unconditional probability
        unconditional_probs = norm_func(unconditional_logits)
        unconditional_idx = unconditional_input_ids
        unconditional_tokens_idx = unconditional_input_ids==pad_token_id
        for i in range(unconditional_idx.shape[0]):
            for j in range(unconditional_idx.shape[1]):
                idx[i,j] = i*unconditional_probs.shape[1]*unconditional_probs.shape[2] + j*unconditional_probs.shape[2] + unconditional_idx[i,j] #Creating indexes for torch.take
        unconditional_selected_vals = torch.take(unconditional_probs,unconditional_idx)
        unconditional_selected_vals[unconditional_tokens_idx] = 1.0
        unconditional_mult_probs = torch.prod(unconditional_selected_vals,axis=1)
        normalised_probs = mult_probs/unconditional_mult_probs #Divide output probs by unconditioning output option probs
    #print(logits,logits.shape)
    #print(probs,probs.shape)
    #print('Input_ids', input_ids,input_ids.shape)
    #print('Token idx',tokens_idx, tokens_idx.shape)
    #print(selected_vals,selected_vals.shape)
    #print(norm_value,norm_value.shape)
    #print(mult_probs,mult_probs.shape)
    arg_max = torch.argmax(normalised_probs)
    #print(normalised_probs,normalised_probs.shape)
    return arg_max

def eval_harness(model, tokenizer, dataset,metadata):
    
    # Define batch sizes
    batch_size = 1
    num_examples = len(dataset)
    num_batches = (num_examples + batch_size - 1) // batch_size
    predictions = []
    corr_count = 0
    print('Num examples ',num_examples)
    
    # Define device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = model.device
    # Evaluate in batches
    for batch_idx in tqdm(range(num_batches)):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_examples)
        batch = dataset[batch_start:batch_end]
        options = [ast.literal_eval(st) for st in batch['options']]
        num_options = len(options[0])
        input_texts = []
        output_texts = []
        
        # Join the prompt with the option
        for e in range(len(batch['wrapped'])):
            if 't5' not in metadata['model']:
                input_texts.extend([f"{batch['wrapped'][e]} {o}" for o in options[e]])
            else:
                input_texts.extend([f"{batch['wrapped'][e]} " for o in options[e]])
                output_texts.extend([f'{o}' for o in options[e]])
        
        #print(output_texts)
        for i in range(batch_size):
            inp = input_texts[(i*num_options):((i+1)*num_options)]
            if 't5' in metadata['model']:
                out = output_texts[(i*num_options):((i+1)*num_options)]
            else:
                out = []
            prompt_input = tokenizer(batch['wrapped'][0]+" ", return_tensors='pt')
            #print(prompt_input.input_ids.shape)
            # Tokenize input texts
            inputs = tokenizer(inp, return_tensors="pt", padding=True, truncation=True).to(device)
            #print(tokenizer.batch_decode(inputs.input_ids))
            #print(inputs.input_ids.shape)
            # Get model outputs
            if 't5' in metadata['model']:
                decoder_input_ids = tokenizer(out, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
                decoder_input_ids = model._shift_right(decoder_input_ids) 
                #print(decoder_input_ids)
                #print(tokenizer.batch_decode(decoder_input_ids))
                #print(inputs.input_ids.shape)
                #print(decoder_input_ids.shape)
                #print(inputs.attention_mask.shape)
                logits = None
                for b in range(inputs.input_ids.shape[0]):
                    #print('Inp', inputs.input_ids[b,:].view(1,-1).shape)
                    #print('Dec', decoder_input_ids[b,:].view(1,-1).shape)
                    #print('Attn mask', inputs.attention_mask.view(1,-1).shape)
                    temp_logits = model(input_ids=(inputs.input_ids[b,:].view(1,-1)).to(device), decoder_input_ids=(decoder_input_ids[b,:].view(1,-1)).to(device), attention_mask=(inputs.attention_mask[b,:].view(1,-1)).to(device)).logits
                    # temp_logits.to(device)
                    if logits==None:
                        logits=temp_logits
                    else:
                        logits = torch.vstack((logits,temp_logits))
                    #print('Logits',logits.shape)
                #logits = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_input_ids, attention_mask=inputs.attention_mask).logits
                #print(logits.shape)
            else:
                # print(inputs.input_ids.shape)
                # logits = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask).logits
                # print(logits.shape)
                logits = None
                for b in range(inputs.input_ids.shape[0]):
                    temp_logits = model(input_ids=(inputs.input_ids[b,:].view(1,-1)).to(device), attention_mask=(inputs.attention_mask[b,:].view(1,-1)).to(device)).logits
                    # temp_logits.to(device)
                    if logits==None:
                        logits=temp_logits
                    else:
                        logits = torch.vstack((logits,temp_logits))
                    #print(logits.shape)    
            
            # Get the prediction
            #print(logits,logits.shape)
            #print(tokenizer.batch_decode(input_ids))
            if 't5' not in metadata['model']:
                input_ids = inputs.input_ids[:,prompt_input.input_ids.shape[1]:]
                logits = logits[:,prompt_input.input_ids.shape[1]-1:-1,:]
            else:
                input_ids = decoder_input_ids[:,1:]
                logits = logits[:,:-1,:]
            #print(logits,logits.shape)
            logits.to("cpu").clone().detach()
            input_ids.to( "cpu" ).clone().detach()
            arg_max = ai_harness_probs(input_ids=input_ids,logits=logits,pad_token_id=tokenizer.pad_token_id).item()
            #print(arg_max,batch['correct answer'][i])

            
            # Append predicted ouput
            if arg_max in range(len(options[i])):
                predictions.append(options[i][arg_max])
            else:
                predictions.append("")
                
            # Print status
            if options[i][arg_max]==batch['correct answer'][i]:
                corr_count+=1
            # if batch_idx%100==0:
            #     print(corr_count,batch_idx+1)
            #     print(corr_count/(batch_idx+1))
                
    return predictions

def eval_generate(model, tokenizer, dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import re
    def generate_text(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        max_length = 5
        outputs = model.generate(inputs["input_ids"], max_new_tokens=max_length)
        generated_text0 = tokenizer.decode(outputs[0][-max_length:], skip_special_tokens=True)
        generated_text = generated_text0.replace(prompt,"").split("\n")[0].strip()
        generated_text = re.sub('[^A-Za-z0-9 ]', "", generated_text).split()
        options = ast.literal_eval(dataset['options'][0])
        num_options = len(options)
        nop = [chr(ord('A')+i) for i in range(num_options)]
        argss = False
        for j in generated_text:
            if j in nop:
                generated_text = j
                argss = True
                break
        if not argss:
            generated_text = ''
        return [generated_text,generated_text0]

    generated_texts = [generate_text(prompt) for prompt in tqdm(dataset['wrapped'])]
    return generated_texts

def calculate_ppa(question, answer_options, model, tokenizer,metadata):
    answer_options = ast.literal_eval(answer_options)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_options = len(answer_options)
    num_agreements = 0
    plurality_answer = None
    max_count = 0
    answer_counts = {option: 0 for option in answer_options}
    lbls_map = {v: k for k, v in tokenizer.vocab.items()}

    for ordering in itertools.permutations(answer_options):
        prompt_text = question + "\nOptions:\n" + "\n".join([f"{chr(65+i)}: {option}" for i, option in enumerate(ordering)]) + "\n" + "Correct option="
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        if 't5' in metadata['model']:
            decoder_input_ids = tokenizer("", return_tensors="pt").input_ids.to(device)
            decoder_input_ids = model._shift_right(decoder_input_ids)
            outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_input_ids, attention_mask=inputs.attention_mask)
        else:
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        logits = outputs.logits[0, -1]
        probs = logits.softmax(dim=-1)
        
        logprobs_dict = {
            lbls_map[i]:
            probs[i].item() for i in range(len(lbls_map))
        }
        
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:200]
        }
        
        logs = {}
        for i,option in enumerate(ordering):
            if chr(65+i) in logprobs_dict:
                if option not in logs:
                    logs[option] = logprobs_dict[chr(65+i)]
                else:
                    logs[option] = max(logs[option],logprobs_dict[chr(65+i)])
            if '▁'+chr(65+i) in logprobs_dict:
                if option not in logs:
                    logs[option] = logprobs_dict['▁'+chr(65+i)]
                else:
                    logs[option] = max(logs[option],logprobs_dict['▁'+chr(65+i)])
            if 'Ġ'+chr(65+i) in logprobs_dict:
                if (chr(65+i)) not in logs:
                    logs[chr(65+i)] = logprobs_dict['Ġ'+chr(65+i)]
                else:
                    logs[chr(65+i)] = max(logs[chr(65+i)],logprobs_dict['Ġ'+chr(65+i)])
        if len(logs)!=0:
            response = max(logs, key=logs.get)
        else:
            response = ""
        #print(logs, response)
        
        if response in answer_counts:
            answer_counts[response] += 1
            if answer_counts[response] > max_count:
                max_count = answer_counts[response]
                plurality_answer = response

    ppa = max_count / math.factorial(num_options)
    #print(ppa)
    return ppa

def eval_mcqa(model,tokenizer,dataset,metadata):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = model.device
    lbls_map = {v: k for k, v in tokenizer.vocab.items()}
    
    def get_ans(prompt_text,options,metadata):
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        if 't5' in metadata['model']:
            decoder_input_ids = tokenizer("", return_tensors="pt").input_ids.to(device)
            decoder_input_ids = model._shift_right(decoder_input_ids)
            outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_input_ids, attention_mask=inputs.attention_mask)
        else:
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        logits = outputs.logits[0, -1]
        probs = logits.softmax(dim=-1)
        
        logprobs_dict = {
            lbls_map[i]:
            probs[i].item() for i in range(len(lbls_map)) #Removed log
        }
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:50]
        }
        #print(logprobs_dict)
        logs = {}
        ordering = ast.literal_eval(options)
        for i,option in enumerate(ordering):
            if chr(65+i) in logprobs_dict:
                if (chr(65+i)) not in logs:
                    logs[chr(65+i)] = logprobs_dict[chr(65+i)]
                else:
                    logs[chr(65+i)] = max(logs[chr(65+i)],logprobs_dict[chr(65+i)])
            if '▁'+chr(65+i) in logprobs_dict:
                if (chr(65+i)) not in logs:
                    logs[chr(65+i)] = logprobs_dict['▁'+chr(65+i)]
                else:
                    logs[chr(65+i)] = max(logs[chr(65+i)],logprobs_dict['▁'+chr(65+i)])
            if 'Ġ'+chr(65+i) in logprobs_dict:
                if (chr(65+i)) not in logs:
                    logs[chr(65+i)] = logprobs_dict['Ġ'+chr(65+i)]
                else:
                    logs[chr(65+i)] = max(logs[chr(65+i)],logprobs_dict['Ġ'+chr(65+i)])
        if len(logs)!=0:
            response = max(logs, key=logs.get)
        else:
            response = ""
        return response
    
    dataset = dataset.to_pandas()
    predictions = []
    for index,row in tqdm(dataset.iterrows()):
        predictions.append(get_ans(row['wrapped'],row['options'],metadata))
    return predictions
    
                
def get_results(model,tokenizer,dataset,metadata):
    # Define output filename
    result_filename = f"{metadata['prompt']}_k{metadata['k']}_run{metadata['run']}_{metadata['eval_method']}.csv"
    result_path = os.path.join(metadata['output_dir'], f"generated_texts/task_{metadata['task']}/{metadata['model']}")
    os.makedirs(result_path,exist_ok=True)
    absolute_result_path = os.path.join(result_path,result_filename)
    
    metric_filename = f"{metadata['prompt']}_k{metadata['k']}_run{metadata['run']}_{metadata['eval_method']}.csv"
    metric_path = os.path.join(metadata['output_dir'],f"metrics/task_{metadata['task']}/{metadata['model']}")
    os.makedirs(metric_path,exist_ok=True)
    absolute_metric_path = os.path.join(metric_path,metric_filename)
    
    if metadata['eval_method'] == 'all':
        # Evaluate using all methods
        harness_predictions = eval_harness(model,tokenizer,dataset,metadata)
        generate_predictions = eval_generate(model,tokenizer,dataset)
        
    elif metadata['eval_method'] == 'harness':
        # Evaluate using ony the harness method
        harness_predictions = eval_harness(model,tokenizer,dataset,metadata)
        
        # Prepare dataframe to save results
        df = dataset.to_pandas()
        df['harness_predictions'] = harness_predictions
        df.drop('wrapped',axis=1,inplace=True)
        
        # Save results
        df.to_csv(absolute_result_path, index=True)

        # Calculate the evaluation metrics
        precision = precision_score(df['correct answer'], df['harness_predictions'], average='macro')
        recall = recall_score(df['correct answer'], df['harness_predictions'], average='macro')
        f1_macro = f1_score(df['correct answer'], df['harness_predictions'], average='macro')
        accuracy = accuracy_score(df['correct answer'], df['harness_predictions'])

        # Print results
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Macro F1: {f1_macro}")
        print(f"Accuracy: {accuracy}")
        
        # Return the results dictionary
        results_dict = {
            'precision':precision,
            'recall':recall,
            'f1_macro':f1_macro,
            'accuracy':accuracy
        } 
        with open(absolute_metric_path, 'w') as json_file:
            json.dump(results_dict, json_file)
        return results_dict
    
    elif metadata['eval_method'] == 'ppa':
        ppa_list = []
        dataset = dataset.to_pandas()
        for index,row in tqdm(dataset.iterrows()):
            ppa_list.append(calculate_ppa(row['wrapped'], row['options'], model, tokenizer,metadata))
        
        ppa_result = {
            'task' : metadata['task'],
            'ppa' : sum(ppa_list)/len(ppa_list),
            'model' : metadata['model']
        }
        with open(absolute_metric_path, 'w') as json_file:
            json.dump(ppa_result, json_file)
        return ppa_result
    
    elif metadata['eval_method'] == 'mcqa':
        generate_predictions = eval_mcqa(model,tokenizer,dataset,metadata)
        df = dataset.to_pandas()
        options = [ast.literal_eval(o) for o in df['options']]
        correct_options = [chr(options[i].index(df['correct answer'][i])+ord('A')) for i in range(len(df['correct answer']))]
        df['correct_options'] = correct_options
        df['generate_predictions'] = generate_predictions
        precision = precision_score(df['correct_options'], df['generate_predictions'], average='macro')
        recall = recall_score(df['correct_options'], df['generate_predictions'], average='macro')
        f1_macro = f1_score(df['correct_options'], df['generate_predictions'], average='macro')
        accuracy = accuracy_score(df['correct_options'], df['generate_predictions'])
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Macro F1: {f1_macro}")
        print(f"Accuracy: {accuracy}")
        results_dict = {
            'precision':precision,
            'recall':recall,
            'f1_macro':f1_macro,
            'accuracy':accuracy
        } 
        
        
        df.to_csv(absolute_result_path, index=True)
        with open(absolute_metric_path, 'w') as json_file:
            json.dump(results_dict, json_file)
        return results_dict
        
    
    else:
        # Evaluate inly using generate method
        generate_predictions = eval_generate(model,tokenizer,dataset)
        df = dataset.to_pandas()
        options = [ast.literal_eval(o) for o in df['options']]
        correct_options = [chr(options[i].index(df['correct answer'][i])+ord('A')) for i in range(len(df['correct answer']))]
        df['correct_options'] = correct_options
        df['generate_predictions'] = [i[0] for i in generate_predictions]
        df['predicted'] = [i[1] for i in generate_predictions]
        precision = precision_score(df['correct_options'], df['generate_predictions'], average='macro')
        recall = recall_score(df['correct_options'], df['generate_predictions'], average='macro')
        f1_macro = f1_score(df['correct_options'], df['generate_predictions'], average='macro')
        accuracy = accuracy_score(df['correct_options'], df['generate_predictions'])
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Macro F1: {f1_macro}")
        print(f"Accuracy: {accuracy}")
        results_dict = {
            'precision':precision,
            'recall':recall,
            'f1_macro':f1_macro,
            'accuracy':accuracy
        } 
        
        
        df.to_csv(absolute_result_path, index=True)
        with open(absolute_metric_path, 'w') as json_file:
            json.dump(results_dict, json_file)
        return results_dict

    
    