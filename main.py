#from pragmatics.eval_tasks.task_15_deixis import eval
import os
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
import torch
from pragmatics.eval_tasks.eval_script import get_results
from pragmatics.eval_tasks.prompt_script import get_data
import pandas as pd
from datasets import Dataset
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.system("echo $CUDA_VISIBLE_DEVICES")
#eval()
from transformers import AutoTokenizer, DataCollatorWithPadding,AutoModelForCausalLM
from transformers import LlamaForCausalLM #llama
from transformers import FalconForCausalLM #Falcon
from transformers import T5ForConditionalGeneration #t5
from transformers import AutoModelForSeq2SeqLM #Flant5
from transformers import GPT2LMHeadModel #gpt2


# Quantisation config for 180b falcon
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

def setup_model(model_name):
    model_paths = {
            'llama-2-7b': '/raid/nlp/models/llama-2-7B-hf',
            'llama-2-7b-chat': '/raid/nlp/models/llama-2-7b-chat-hf',
            'llama-2-13b': '/raid/nlp/models/llama-2-13b-hf',
            'llama-2-13b-chat': '/raid/nlp/models/llama-2-13b-chat-hf',
            'llama-2-70b': '/raid/nlp/models/llama-2-70b-hf',
            'llama-2-70b-chat': '/raid/nlp/models/llama-2-70b-chat-hf',
            't5-3b': '/raid/nlp/models/t5-3b',
            't5-11b': '/raid/nlp/models/t5-11b',
            'flan-t5-small': '/raid/nlp/models/flan-t5-small',
            'flan-t5-base': '/raid/nlp/models/flan-t5-base',
            'flan-t5-large': '/raid/nlp/models/flan-t5-large',
            'flan-t5-xl': '/raid/nlp/models/flan-t5-xl',
            'flan-t5-xxl': '/raid/nlp/models/flan-t5-xxl',
            'falcon-40b-instruct': '/raid/nlp/models/falcon-40b-instruct',
            'falcon-40b': '/raid/nlp/models/falcon-40b',
            'gpt2': '/raid/nlp/models/gpt2',
            'phi-1': '/raid/nlp/models/phi-1',
            'falcon-7b-instruct': '/raid/nlp/models/falcon-7b-instruct',
            'falcon-7b': '/raid/nlp/models/falcon-7b',
            'falcon-180b': '/raid/nlp/models/falcon-180b/falcon-180B',
            'phi-1_5': '/raid/nlp/models/phi-1_5'
        }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name not in model_paths.keys():
        raise Exception('Model not found')
    model = None
    tokenizer = None
    torch.cuda.empty_cache()
    print('Loading Model...')
    tokenizer = AutoTokenizer.from_pretrained(model_paths[model_name])
    if model_name=='llama-2-7b':
        model = LlamaForCausalLM.from_pretrained(model_paths[model_name], device_map="auto", torch_dtype=torch.float16)
        model.bfloat16()
    elif model_name=='llama-2-7b-chat':
        model = LlamaForCausalLM.from_pretrained(model_paths[model_name], device_map="auto", torch_dtype=torch.float16)
        model.bfloat16()
    elif model_name=='llama-2-13b':
        model = LlamaForCausalLM.from_pretrained(model_paths[model_name], device_map="auto", torch_dtype=torch.float16)
        model.bfloat16()
    elif model_name=='llama-2-13b-chat':
        model = LlamaForCausalLM.from_pretrained(model_paths[model_name], device_map="auto", torch_dtype=torch.float16)
        model.bfloat16()
    elif model_name=='llama-2-70b':
        model = LlamaForCausalLM.from_pretrained(model_paths[model_name], device_map="auto", load_in_8bit=True)
        model.bfloat16()
    elif model_name=='llama-2-70b-chat':
        model = LlamaForCausalLM.from_pretrained(model_paths[model_name], device_map="auto", load_in_8bit=True)
        model.bfloat16()
    elif model_name=='t5-3b':
        model = T5ForConditionalGeneration.from_pretrained(model_paths[model_name],device_map="auto", torch_dtype=torch.float16)
    elif model_name=='t5-11b':
        model = T5ForConditionalGeneration.from_pretrained(model_paths[model_name], device_map="auto", torch_dtype=torch.float16)
    elif model_name=='flan-t5-small':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_paths[model_name],device_map="auto", torch_dtype=torch.float16)
    elif model_name=='flan-t5-base':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_paths[model_name],device_map="auto", torch_dtype=torch.float16)
    elif model_name=='flan-t5-large':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_paths[model_name],device_map="auto", torch_dtype=torch.float16)
    elif model_name=='flan-t5-xl':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_paths[model_name],device_map="auto", torch_dtype=torch.float16)
    elif model_name=='flan-t5-xxl':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_paths[model_name],device_map="auto", torch_dtype=torch.float16)
    elif model_name=='gpt2':
        model = AutoModelForCausalLM.from_pretrained(model_paths[model_name],device_map="auto", torch_dtype=torch.float16)
    elif model_name=='falcon-7b-instruct':
        model = AutoModelForCausalLM.from_pretrained(model_paths[model_name],device_map="auto",trust_remote_code=True, torch_dtype=torch.float16)
    elif model_name=='falcon-7b':
        model = AutoModelForCausalLM.from_pretrained(model_paths[model_name], device_map="auto",trust_remote_code=True,torch_dtype=torch.float16)
    elif model_name=='falcon-40b-instruct':
        model = AutoModelForCausalLM.from_pretrained(model_paths[model_name], device_map="auto",trust_remote_code=True,load_in_8bit=True)
    elif model_name=='falcon-40b':
        model = AutoModelForCausalLM.from_pretrained(model_paths[model_name], device_map="auto",trust_remote_code=True,load_in_8bit=True)
    elif model_name=='falcon-180b':
        model = AutoModelForCausalLM.from_pretrained(
            model_paths[model_name],
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model.config.use_cache = False
    elif model_name=='phi-1':
        model = AutoModelForCausalLM.from_pretrained(model_paths[model_name],trust_remote_code=True, torch_dtype=torch.float16)
        model.to(device)
    elif model_name=='phi-1_5':
        model = AutoModelForCausalLM.from_pretrained(model_paths[model_name],trust_remote_code=True, torch_dtype=torch.float16)
        model.to(device)
    else:
        raise Exception('Model not in list')
    model.eval()
    print('Loaded')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    return model,tokenizer

if __name__=="__main__":
    metadata= {
        'task':0,
        'model':'gpt2',
        'input_dir':'./pragmatics/global_datasets/',
        'model_path':'/raid/nlp/models/llama-2-13b-chat-hf/',
        'prompt':'zero_shot', # zero_shot to few_shot
        'run':1,
        'output_dir':'/raid/nlp/pranavg/iclr/Results',
        'eval_method':'harness', #mcqa or harness or ppa
        'k':0 # 0,3,5
    }
    models_small = ['t5-3b','flan-t5-xl','flan-t5-small','flan-t5-base','flan-t5-large']
    models_medium = ['flan-t5-small','flan-t5-base','flan-t5-large','phi-1','phi-1_5','t5-3b','flan-t5-xl']
    models_large = ['falcon-40b','falcon-40b-instruct','llama-2-70b','llama-2-70b-chat']
    models_all = ['falcon-7b-instruct','t5-11b','flan-t5-xxl','falcon-40b','falcon-40b-instruct','llama-2-70b','llama-2-70b-chat']
    # models = ['llama-2-7b','llama-2-7b-chat','falcon-7b-instruct','falcon-7b']
    #models = ['falcon-7b']
    models = ['llama-2-70b','llama-2-70b-chat','llama-2-7b','llama-2-7b-chat','llama-2-13b','llama-2-13b-chat','t5-11b']
    # models = ['llama-2-70b-chat']
    # models = ['flan-t5-small']
    #models = ['flan-t5-large','flan-t5-xl','flan-t5-xxl']
    # models = ['llama-2-13b-chat','flan-t5-xxl','t5-11b']
    # models = models_all
    #tasks = [1,4,5,6,7,8,14,11,3]
    tasks = [16,17,18,19]
    #models = ['gpt2']
    #li = [14]
    #li = ['llama-2-7b-chat']
    #li = ['phi-1','phi-1_5','t5-3b','flan-t5-xl','llama-2-7b','llama-2-7b-chat','t5-11b','flan-t5-xxl','llama-2-13b','llama-2-13b-chat','falcon-7b-instruct']
    #li = ['flan-t5-xl','llama-2-7b','llama-2-7b-chat','t5-11b','flan-t5-xxl','llama-2-13b','llama-2-13b-chat','falcon-7b-instruct']
    for m in models:
        metadata['model'] = m
        model,tokenizer = setup_model(metadata['model'])
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
            dataset = Dataset.from_pandas(dataset)
            result_dict = get_results(model,tokenizer,dataset,metadata)
            print("-------")
            print(dataset['wrapped'][0])
            print(result_dict)
            print("-----")

        
    
    
    