import ast
import pandas as pd
import json

def human_eval(row,question):
    options = ast.literal_eval(row['options'])
    context = row['pretext'].strip('\n')
    text = f"{question}\n{context}"
    opts = []
    for i in range(len(options)):
        opts.append({"value":options[i]})
    return {
        "data":{
            "text": text,
            "options":opts
        }
    }
    
# Subsample the dataset with equal contribution for all classes
def subsampled_set(gdata,dev_len=101):
    if len(set(gdata['correct answer']))>5:
        devset = gdata.sample(dev_len, random_state=42)
        sampled_rows = devset.index
        gdata.drop(sampled_rows, inplace=True)
        gdata.reset_index(drop=True, inplace=True)
        devset.reset_index(drop=True, inplace=True)
    else:
        devset = pd.DataFrame()
        num_options = len(set(gdata['correct answer']))
        sampled_indices = []
        for cat in gdata['correct answer'].unique():
            cat_df = gdata[gdata['correct answer'] == cat]
            print(cat)
            sampled_cat_df = cat_df.sample(n = dev_len//num_options, random_state=42)
            devset = pd.concat([devset,sampled_cat_df])
            sampled_indices.extend(sampled_cat_df.index)
        gdata.drop(sampled_indices, inplace=True)
        gdata.reset_index(drop=True, inplace=True)
        devset.reset_index(drop=True, inplace=True)
    return devset,gdata

# Convert each question into zero shot prompt
def zero_shot(row,question,metadata):
    options = ast.literal_eval(row['options'])
    context = row['pretext'].strip('\n')
    if metadata['eval_method'] == 'ppa':
        wrapped = f"{question}\n{context}"
    elif metadata['eval_method'] == 'harness':
        wrapped = f"{question}\n{context}\nCorrect Answer="
    else:  
        wrapped = f"{question}\n{context}\nOptions:\n"  
        for i in range(len(options)):
            wrapped += f"{chr(ord('A')+i)}: {options[i]}\n"
        wrapped += f"Correct option= " 
    return wrapped

# Similar to zero shot but do not append question/instruction
def zero_shot_f(row,metadata):
    options = ast.literal_eval(row['options'])
    context = row['pretext'].strip('\n')
    if metadata['eval_method'] == 'harness':
        wrapped = f"{context}\nCorrect Answer="
    else:
        wrapped = f"{context}\nOptions:\n"
        for i in range(len(options)):
            wrapped += f"{chr(ord('A')+i)}: {options[i]}\n"
        wrapped += f"Correct option= " 
    return wrapped
    
def dev_set(gdata,metadata,dev_len=20):
    devset,gdata = subsampled_set(gdata,dev_len)
    option_list = [ast.literal_eval(o) for o in devset['options']]
    if metadata['eval_method'] == 'harness':
        dev_wrapped = [f"{devset['wrapped_zshot'][r]} {devset['correct answer'][r]}" for r in range(len(devset))]
    else:
        dev_wrapped = [f"{devset['wrapped_zshot'][r]}{chr(ord('A')+option_list[r].index(devset['correct answer'][r]))}: {devset['correct answer'][r]}" for r in range(len(devset))]
    devset['wrapped_zshot'] = dev_wrapped
    devset.reset_index(drop=True, inplace=True)
    return gdata,devset
            

def few_shot(row,devset,metadata,instruction):
    sampled = devset.sample(metadata['k'])
    wrapped = "\n\n".join(sampled['wrapped_zshot'])
    return instruction + "\n" + wrapped + "\n" + zero_shot(row,"",metadata)
                                                              
# Task 0
def direct_indirect(gdata,metadata):
    wrapped = ""
    question = "Your task is to label the 'Response' as an Indirect or Direct answer based on the Context and Question:\n"
    prompt = metadata['prompt']
    
    if prompt == 'zero_shot':
        gdata['wrapped'] = gdata.apply(lambda row: zero_shot(row,question,metadata), axis=1)
        return gdata
    
    elif prompt == 'few_shot':
        instruction = question
        gdata['wrapped_zshot'] = gdata.apply(lambda row: zero_shot_f(row,metadata), axis=1)
        gdata, devset = dev_set(gdata,metadata)
        gdata['wrapped'] = gdata.apply(lambda row: few_shot(row,devset,metadata,instruction), axis=1)
        return gdata
    
    elif prompt == 'human':
        human_list = []
        human_devset,gdata = subsampled_set(gdata)
        
        for index,row in human_devset.iterrows():
            human_list.append(human_eval(row,question))
        return human_list 

# Task 1
def response_classification(gdata,metadata):
    wrapped = ""
    question = "Your task is to interpret Y's answer to X's question into one of the options:\nA: Yes\nB: No\nC: Yes, subject to some conditions\nD: In the middle, neither yes nor no\nE: Other\n"
    prompt = metadata['prompt']
    
    if prompt == 'zero_shot':
        gdata['wrapped'] = gdata.apply(lambda row: zero_shot(row,question,metadata), axis=1)
        return gdata
    
    elif prompt == 'few_shot':
        instruction = question
        gdata['wrapped_zshot'] = gdata.apply(lambda row: zero_shot_f(row,metadata), axis=1)
        gdata, devset = dev_set(gdata,metadata)
        gdata['wrapped'] = gdata.apply(lambda row: few_shot(row,devset,metadata,instruction), axis=1)
        return gdata
    
    elif prompt == 'human':
        human_list = []
        human_devset,gdata = subsampled_set(gdata)
        
        for index,row in human_devset.iterrows():
            human_list.append(human_eval(row,question))
        return human_list 

# Task 2
def response_classification_with_meaning(gdata,metadata):
    wrapped = ""
    question = "Your task is to interpret Y's answer to X's question into one of the options:\nA: Yes\nB: No\nC: Yes, subject to some conditions\nD: In the middle, neither yes nor no\nE: Other\n"
    prompt = metadata['prompt']
    
    if prompt == 'zero_shot':
        gdata['wrapped'] = gdata.apply(lambda row: zero_shot(row,question,metadata), axis=1)
        return gdata
    
    elif prompt == 'few_shot':
        instruction = question
        gdata['wrapped_zshot'] = gdata.apply(lambda row: zero_shot_f(row,metadata), axis=1)
        gdata, devset = dev_set(gdata,metadata)
        gdata['wrapped'] = gdata.apply(lambda row: few_shot(row,devset,metadata,instruction), axis=1)
        return gdata
    
    elif prompt == 'human':
        human_list = []
        human_devset,gdata = subsampled_set(gdata)
        
        for index,row in human_devset.iterrows():
            human_list.append(human_eval(row,question))
        return human_list 

# Task 3
def implicature_recovery(gdata,metadata):
    wrapped = ""
    question = "Your task is to understand the implied meaning in Speaker_2's last response and give the explicit meaning:\n"
    #question = "Your task is to recover the meaning of the implcature from speaker 2's last response from a conversation"
    prompt = metadata['prompt']
    
    if prompt == 'zero_shot':
        gdata['wrapped'] = gdata.apply(lambda row: zero_shot(row,question,metadata), axis=1)
        return gdata
    
    elif prompt == 'few_shot':
        instruction = question
        gdata['wrapped_zshot'] = gdata.apply(lambda row: zero_shot_f(row,metadata), axis=1)
        gdata, devset = dev_set(gdata,metadata)
        gdata['wrapped'] = gdata.apply(lambda row: few_shot(row,devset,metadata,instruction), axis=1)
        return gdata
    
    elif prompt == 'human':
        human_list = []
        human_devset,gdata = subsampled_set(gdata)
        
        for index,row in human_devset.iterrows():
            human_list.append(human_eval(row,question))
        return human_list  

# Task 4
def figurative_agreement(gdata,metadata):
    wrapped = ""
    question = "Your task is to decide if Speaker_2 Agrees or Disagrees with Speaker_1 in the conversation:\n"
    
    prompt = metadata['prompt']
    
    if prompt == 'zero_shot':
        gdata['wrapped'] = gdata.apply(lambda row: zero_shot(row,question,metadata), axis=1)
        return gdata
    
    elif prompt == 'few_shot':
        instruction = question
        gdata['wrapped_zshot'] = gdata.apply(lambda row: zero_shot_f(row,metadata), axis=1)
        gdata, devset = dev_set(gdata,metadata)
        gdata['wrapped'] = gdata.apply(lambda row: few_shot(row,devset,metadata,instruction), axis=1)
        return gdata
    
    elif prompt == 'human':
        human_list = []
        human_devset,gdata = subsampled_set(gdata)
        
        for index,row in human_devset.iterrows():
            human_list.append(human_eval(row,question))
        return human_list  

# Task 5
def figurative_sarcasm(gdata,metadata):
    wrapped = ""
    question = "Your task is to decide if Speaker_2 Agrees or is being Sarcastic with Speaker_1 in the conversation:\n"
    prompt = metadata['prompt']
    
    if prompt == 'zero_shot':
        gdata['wrapped'] = gdata.apply(lambda row: zero_shot(row,question,metadata), axis=1)
        return gdata
    
    elif prompt == 'few_shot':
        instruction = question
        gdata['wrapped_zshot'] = gdata.apply(lambda row: zero_shot_f(row,metadata), axis=1)
        gdata, devset = dev_set(gdata,metadata)
        gdata['wrapped'] = gdata.apply(lambda row: few_shot(row,devset,metadata,instruction), axis=1)
        return gdata
    
    elif prompt == 'human':
        human_list = []
        human_devset,gdata = subsampled_set(gdata)
        
        for index,row in human_devset.iterrows():
            human_list.append(human_eval(row,question))
        return human_list  

# Task 6
def figurative_reasoning(gdata,metadata):
    wrapped = ""
    question = "Your task is to identify the correct meaning of the figurative sentence:\n"
    prompt = metadata['prompt']
    
    if prompt == 'zero_shot':
        gdata['wrapped'] = gdata.apply(lambda row: zero_shot(row,question,metadata), axis=1)
        return gdata
    
    elif prompt == 'few_shot':
        instruction = question
        gdata['wrapped_zshot'] = gdata.apply(lambda row: zero_shot_f(row,metadata), axis=1)
        gdata, devset = dev_set(gdata,metadata)
        gdata['wrapped'] = gdata.apply(lambda row: few_shot(row,devset,metadata,instruction), axis=1)
        return gdata
    
    elif prompt == 'human':
        human_list = []
        human_devset,gdata = subsampled_set(gdata)
        
        for index,row in human_devset.iterrows():
            human_list.append(human_eval(row,question))
        return human_list  

# Task 7
def positive_figurative_reasoning(gdata,metadata):
    wrapped = ""
    question = "Your task is to identify the correct meaning of the figurative sentence from the given hint:\n"
    prompt = metadata['prompt']
    
    if prompt == 'zero_shot':
        gdata['wrapped'] = gdata.apply(lambda row: zero_shot(row,question,metadata), axis=1)
        return gdata
    
    elif prompt == 'few_shot':
        instruction = question
        gdata['wrapped_zshot'] = gdata.apply(lambda row: zero_shot_f(row,metadata), axis=1)
        gdata, devset = dev_set(gdata,metadata)
        gdata['wrapped'] = gdata.apply(lambda row: few_shot(row,devset,metadata,instruction), axis=1)
        return gdata
    
    elif prompt == 'human':
        human_list = []
        human_devset,gdata = subsampled_set(gdata)
        
        for index,row in human_devset.iterrows():
            human_list.append(human_eval(row,question))
        return human_list 

def contrastive_figurative_reasoning(gdata,metadata):
    wrapped = ""
    question = "Your task is to identify the correct meaning of the figurative sentence from the given hint:\n"
    prompt = metadata['prompt']
    
    if prompt == 'zero_shot':
        gdata['wrapped'] = gdata.apply(lambda row: zero_shot(row,question,metadata), axis=1)
        return gdata
    
    elif prompt == 'few_shot':
        instruction = question
        gdata['wrapped_zshot'] = gdata.apply(lambda row: zero_shot_f(row,metadata), axis=1)
        gdata, devset = dev_set(gdata,metadata)
        gdata['wrapped'] = gdata.apply(lambda row: few_shot(row,devset,metadata,instruction), axis=1)
        return gdata
    
    elif prompt == 'human':
        human_list = []
        human_devset,gdata = subsampled_set(gdata)
        
        for index,row in human_devset.iterrows():
            human_list.append(human_eval(row,question))
        return human_list  

def implicature_nli(gdata,metadata):
    wrapped = ""
    question = ""
    prompt = metadata['prompt']
    
    if prompt == 'zero_shot':
        gdata['wrapped'] = gdata.apply(lambda row: zero_shot(row,question,metadata), axis=1)
        return gdata
    
    elif prompt == 'few_shot':
        instruction = question
        gdata['wrapped_zshot'] = gdata.apply(lambda row: zero_shot_f(row,metadata), axis=1)
        gdata, devset = dev_set(gdata,metadata)
        gdata['wrapped'] = gdata.apply(lambda row: few_shot(row,devset,metadata,instruction), axis=1)
        return gdata
    
    elif prompt == 'human':
        human_list = []
        human_devset,gdata = subsampled_set(gdata)
        
        for index,row in human_devset.iterrows():
            human_list.append(human_eval(row,question))
        return human_list 

#Task 11
def deixis(gdata,metadata):
    wrapped = ""
    question = "Your task is to answer the given question based on the conversation:\n"
    prompt = metadata['prompt']
    
    if prompt == 'zero_shot':
        gdata['wrapped'] = gdata.apply(lambda row: zero_shot(row,question,metadata), axis=1)
        return gdata
    
    elif prompt == 'few_shot':
        instruction = question
        gdata['wrapped_zshot'] = gdata.apply(lambda row: zero_shot_f(row,metadata), axis=1)
        gdata, devset = dev_set(gdata,metadata)
        gdata['wrapped'] = gdata.apply(lambda row: few_shot(row,devset,metadata,instruction), axis=1)
        return gdata
    
    elif prompt == 'human':
        human_list = []
        human_devset,gdata = subsampled_set(gdata)
        
        for index,row in human_devset.iterrows():
            human_list.append(human_eval(row,question))
        return human_list  

#Task 10
def presupposition_nli(gdata,metadata):
    wrapped = ""
    question = ""
    prompt = metadata['prompt']
    
    if prompt == 'zero_shot':
        gdata['wrapped'] = gdata.apply(lambda row: zero_shot(row,question,metadata), axis=1)
        return gdata
    
    elif prompt == 'few_shot':
        instruction = question
        gdata['wrapped_zshot'] = gdata.apply(lambda row: zero_shot_f(row,metadata), axis=1)
        gdata, devset = dev_set(gdata,metadata)
        gdata['wrapped'] = gdata.apply(lambda row: few_shot(row,devset,metadata,instruction), axis=1)
        return gdata
    
    elif prompt == 'human':
        human_list = []
        human_devset,gdata = subsampled_set(gdata)
        
        for index,row in human_devset.iterrows():
            human_list.append(human_eval(row,question))
        return human_list 
#Task 12
def implicature_cancellability(gdata,metadata):
    pass

#Task 13
def presupposition_qa(gdata,metadata):
    wrapped = ""
    question = "Your task is to deduce if the Assumption is valid or invalid based on the conversation:\n"
    prompt = metadata['prompt']
    
    if prompt == 'zero_shot':
        gdata['wrapped'] = gdata.apply(lambda row: zero_shot(row,question,metadata), axis=1)
        return gdata
    
    elif prompt == 'few_shot':
        instruction = question
        gdata['wrapped_zshot'] = gdata.apply(lambda row: zero_shot_f(row,metadata), axis=1)
        gdata, devset = dev_set(gdata,metadata)
        gdata['wrapped'] = gdata.apply(lambda row: few_shot(row,devset,metadata,instruction), axis=1)
        return gdata
    
    elif prompt == 'human':
        human_list = []
        human_devset,gdata = subsampled_set(gdata)
        
        for index,row in human_devset.iterrows():
            human_list.append(human_eval(row,question))
        return human_list  

#Task 14
def metonymy(gdata,metadata):
    wrapped = ""
    question = "Your task is to answer the Question based on the given Context:\n"
    prompt = metadata['prompt']
    
    if prompt == 'zero_shot':
        gdata['wrapped'] = gdata.apply(lambda row: zero_shot(row,question,metadata), axis=1)
        return gdata
    
    elif prompt == 'few_shot':
        instruction = question
        gdata['wrapped_zshot'] = gdata.apply(lambda row: zero_shot_f(row,metadata), axis=1)
        gdata, devset = dev_set(gdata,metadata)
        gdata['wrapped'] = gdata.apply(lambda row: few_shot(row,devset,metadata,instruction), axis=1)
        return gdata
    
    elif prompt == 'human':
        human_list = []
        human_devset,gdata = subsampled_set(gdata)
        
        for index,row in human_devset.iterrows():
            human_list.append(human_eval(row,question))
        return human_list 

#Task 15
def esd(gdata,metadata):
    wrapped = ""
    question = "Your task is to answer the Question based on the given Conservation\n"
    prompt = metadata['prompt']
    
    if prompt == 'zero_shot':
        gdata['wrapped'] = gdata.apply(lambda row: zero_shot(row,question,metadata), axis=1)
        return gdata
    
    elif prompt == 'few_shot':
        instruction = question
        gdata['wrapped_zshot'] = gdata.apply(lambda row: zero_shot_f(row,metadata), axis=1)
        gdata, devset = dev_set(gdata,metadata)
        gdata['wrapped'] = gdata.apply(lambda row: few_shot(row,devset,metadata,instruction), axis=1)
        return gdata
    
    elif prompt == 'human':
        human_list = []
        human_devset,gdata = subsampled_set(gdata)
        
        for index,row in human_devset.iterrows():
            human_list.append(human_eval(row,question))
        return human_list 
#Task 16
def nope(gdata,metadata):
    wrapped = ""
    question = ""
    prompt = metadata['prompt']
    
    if prompt == 'zero_shot':
        gdata['wrapped'] = gdata.apply(lambda row: zero_shot(row,question,metadata), axis=1)
        return gdata
    
    elif prompt == 'few_shot':
        instruction = question
        gdata['wrapped_zshot'] = gdata.apply(lambda row: zero_shot_f(row,metadata), axis=1)
        gdata, devset = dev_set(gdata,metadata)
        gdata['wrapped'] = gdata.apply(lambda row: few_shot(row,devset,metadata,instruction), axis=1)
        return gdata
    
    elif prompt == 'human':
        human_list = []
        human_devset,gdata = subsampled_set(gdata)
        
        for index,row in human_devset.iterrows():
            human_list.append(human_eval(row,question))
        return human_list 
#Task 17
def ddeixis(gdata,metadata):
    wrapped = ""
    question = "Your task is to answer the given question based on the conversation:\n"
    prompt = metadata['prompt']
    
    if prompt == 'zero_shot':
        gdata['wrapped'] = gdata.apply(lambda row: zero_shot(row,question,metadata), axis=1)
        return gdata
    
    elif prompt == 'few_shot':
        instruction = question
        gdata['wrapped_zshot'] = gdata.apply(lambda row: zero_shot_f(row,metadata), axis=1)
        gdata, devset = dev_set(gdata,metadata)
        gdata['wrapped'] = gdata.apply(lambda row: few_shot(row,devset,metadata,instruction), axis=1)
        return gdata
    
    elif prompt == 'human':
        human_list = []
        human_devset,gdata = subsampled_set(gdata)
        
        for index,row in human_devset.iterrows():
            human_list.append(human_eval(row,question))
        return human_list  
#Task 18
def sdeixis(gdata,metadata):
    wrapped = ""
    question = "Your task is to answer the given question based on the conversation:\n"
    prompt = metadata['prompt']
    
    if prompt == 'zero_shot':
        gdata['wrapped'] = gdata.apply(lambda row: zero_shot(row,question,metadata), axis=1)
        return gdata
    
    elif prompt == 'few_shot':
        instruction = question
        gdata['wrapped_zshot'] = gdata.apply(lambda row: zero_shot_f(row,metadata), axis=1)
        gdata, devset = dev_set(gdata,metadata)
        gdata['wrapped'] = gdata.apply(lambda row: few_shot(row,devset,metadata,instruction), axis=1)
        return gdata
    
    elif prompt == 'human':
        human_list = []
        human_devset,gdata = subsampled_set(gdata)
        
        for index,row in human_devset.iterrows():
            human_list.append(human_eval(row,question))
        return human_list  
#Task 19
def tdeixis(gdata,metadata):
    wrapped = ""
    question = "Your task is to answer the given question based on the conversation:\n"
    prompt = metadata['prompt']
    
    if prompt == 'zero_shot':
        gdata['wrapped'] = gdata.apply(lambda row: zero_shot(row,question,metadata), axis=1)
        return gdata
    
    elif prompt == 'few_shot':
        instruction = question
        gdata['wrapped_zshot'] = gdata.apply(lambda row: zero_shot_f(row,metadata), axis=1)
        gdata, devset = dev_set(gdata,metadata)
        gdata['wrapped'] = gdata.apply(lambda row: few_shot(row,devset,metadata,instruction), axis=1)
        return gdata
    
    elif prompt == 'human':
        human_list = []
        human_devset,gdata = subsampled_set(gdata)
        
        for index,row in human_devset.iterrows():
            human_list.append(human_eval(row,question))
        return human_list  

task_mapper = {
    0 : direct_indirect,
    1 : response_classification,
    2 : response_classification_with_meaning,
    3 : implicature_recovery,
    4 : figurative_agreement,
    5 : figurative_sarcasm,
    6 : figurative_reasoning,
    7 : positive_figurative_reasoning,
    8 : contrastive_figurative_reasoning,
    9 : implicature_nli,
    10 : presupposition_nli,
    11 : deixis,
    12 : implicature_cancellability,
    13 : presupposition_qa,
    14 : metonymy,
    15 : esd,
    16: nope,
    17: ddeixis,
    18: sdeixis,
    19: tdeixis
}

def get_data(gdata,metadata):
    task = metadata['task']
    prompt = metadata['prompt']
    if prompt=='human':
        human_list = task_mapper[task](gdata,metadata)
        print(human_list[0])
        json_filename = f"./pragmatics/human_data/task_{task}_human_dataset.json"
        with open(json_filename, 'w') as json_file:
            json.dump(human_list, json_file)
        prompt = "zero_shot"
        gdata = task_mapper[task](gdata,metadata) 
    else:
        gdata = task_mapper[task](gdata,metadata)
        if metadata['eval_method'] == 'ppa':
            ppa_data,new_gdata = subsampled_set(gdata,dev_len=50)  
            return ppa_data
    return gdata