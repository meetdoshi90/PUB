import pandas as pd
from datasets import Dataset
from sklearn.utils import shuffle
import csv

df = pd.read_csv('task_11.csv')
print(len(df))