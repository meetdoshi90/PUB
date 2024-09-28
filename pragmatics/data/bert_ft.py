from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
import torch
from transformers import get_scheduler
from tqdm.auto import tqdm
import numpy as np
from datasets import load_metric
import pandas as pd
import datasets

df = pd.read_csv('circa-data_impl_type.csv')


def tokenization(example):
    return tokenizer(example["question-X"],example["answer-Y"], padding= 'longest')