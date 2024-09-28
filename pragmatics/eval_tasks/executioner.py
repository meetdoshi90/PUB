import torch
from transformers import AutoTokenizer, DataCollatorWithPadding,AutoModelForCausalLM
from transformers import LlamaForCausalLM #llama
from transformers import FalconForCausalLM #Falcon
from transformers import T5ForConditionalGeneration #t5
from transformers import AutoModelForSeq2SeqLM #Flant5
from transformers import GPT2LMHeadModel #gpt2
import os
import torch
import time

class Executioner():
    def __init__(self,model='gpt2'):
        self.model_path = {
            'llama-2-7b': '/raid/nlp/pranavg/iclr/pragmatics/models/llama-2-7B-hf',
            'llama-2-7b-chat': '/raid/nlp/pranavg/iclr/pragmatics/models/llama-2-7b-chat-hf',
            'llama-2-13b': '/raid/nlp/pranavg/iclr/pragmatics/models/llama-2-13b-hf',
            'llama-2-13b-chat': '/raid/nlp/pranavg/iclr/pragmatics/models/llama-2-13b-chat-hf',
            'llama-2-70b': '/raid/nlp/pranavg/iclr/pragmatics/models/llama-2-70b-hf',
            'llama-2-70b-chat': '/raid/nlp/pranavg/iclr/pragmatics/models/llama-2-70b-chat-hf',
            't5-3b': '/raid/nlp/pranavg/iclr/pragmatics/models/t5-3b',
            't5-11b': '/raid/nlp/pranavg/iclr/pragmatics/models/t5-11b',
            'flan-t5-xl': '/raid/nlp/pranavg/iclr/pragmatics/models/flan-t5-xl',
            'flan-t5-xxl': '/raid/nlp/pranavg/iclr/pragmatics/models/flan-t5-xxl',
            'falcon-40b-instruct': '/raid/nlp/pranavg/iclr/pragmatics/models/falcon-40b-instruct',
            'falcon-40b': '/raid/nlp/pranavg/iclr/pragmatics/models/falcon-40b',
            'falcon-7b-instruct': '/raid/nlp/pranavg/iclr/pragmatics/models/falcon-7b-instruct',
            'falcon-7b': '/raid/nlp/pranavg/iclr/pragmatics/models/falcon-7b',
            'gpt2': '/raid/nlp/pranavg/iclr/pragmatics/models/gpt2'
        }
        self.model = None
        self.tokenizer = None
        if model not in self.model_path:
            raise Exception('Model not found')
        #utilisation = self.gpuStat()
        torch.cuda.empty_cache()
        #gpu_num = torch.argmax(torch.tensor(utilisation))
        #self.device = 'cuda:'+str(gpu_num.item())
        self.device = 'cuda:0'
        if model=='llama-2-7b':
            self.model = LlamaForCausalLM.from_pretrained(self.model_path[model]).to(self.device)
        elif model=='llama-2-7b-chat':
            self.model = LlamaForCausalLM.from_pretrained(self.model_path[model], device_map="auto", torch_dtype=torch.float16)
        elif model=='llama-2-13b':
            self.model = LlamaForCausalLM.from_pretrained(self.model_path[model], device_map="auto", torch_dtype=torch.float16)
        elif model=='llama-2-13b-chat':
            self.model = LlamaForCausalLM.from_pretrained(self.model_path[model], device_map="auto", torch_dtype=torch.float16)
        elif model=='llama-2-70b':
            self.model = LlamaForCausalLM.from_pretrained(self.model_path[model], device_map="auto", torch_dtype=torch.float16)
        elif model=='llama-2-70b-chat':
            self.model = LlamaForCausalLM.from_pretrained(self.model_path[model], device_map="auto", torch_dtype=torch.float16)
        elif model=='t5-3b':
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path[model], device_map="auto", torch_dtype=torch.float16)
        elif model=='t5-11b':
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path[model], device_map="auto", torch_dtype=torch.float16)
        elif model=='flan-t5-xl':
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path[model], device_map="auto", torch_dtype=torch.float16)
        elif model=='flan-t5-xxl':
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path[model], device_map="auto", torch_dtype=torch.float16)
        elif model=='gpt2':
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path[model], device_map="auto", torch_dtype=torch.float16)
        elif model=='falcon-40b-instruct':
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path[model], device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
        elif model=='falcon-7b-instruct':
            self.model = FalconForCausalLM.from_pretrained(self.model_path[model], device_map="auto", torch_dtype=torch.float16)
        elif model=='falcon-7b':
            self.model = FalconForCausalLM.from_pretrained(self.model_path[model], device_map="auto", torch_dtype=torch.float16)
        elif model=='falcon-40b':
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path[model], device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
        else:
            raise Exception('Model not in list')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path[model])
        self.model.eval()
        print(f'{model} loaded....')
    
    @staticmethod
    def gpuStat(id=None):
        total_memory_available = []
        if id == None:
            for i in range(8): #8 Nvidia A100 GPUs
                stat = torch.cuda.mem_get_info(i)
                #print(stat)
                free = stat[0]
                total_memory_available.append(free/(1024**3))
        else:
            stat = torch.cuda.mem_get_info(id)
            #print(stat)
            free = stat[0]
            total_memory_available.append(free/(1024**3))
        return total_memory_available


    def getModel(self):
        return self.model
    
    def getTokenizer(self):
        return self.tokenizer
        
if __name__=='__main__':
    e = Executioner('falcon-7b')
    model = e.model
    tokenizer = e.tokenizer
    device = e.device
    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    #print(e.gpuStat())
