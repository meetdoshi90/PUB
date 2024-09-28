import pandas as pd
from datasets import Dataset
from sklearn.utils import shuffle

def read_data(data_path):
    df = pd.read_csv(data_path)
    df = df.dropna()
    df = df[df['answer'].isin(['yes','no'])].reset_index(drop=True)
    return df
# context,question,answer

def prepare_dataset(df):
    df = shuffle(df)
    df = df[:1000]
    options = ['yes','no']
    df['pretext'] = df.apply(lambda x: f"Conversation:\n{x['context']}Question: {x['question']}?\n", axis=1)
    df['options'] = [options] * len(df)
    df.rename(columns={'answer': 'correct answer'}, inplace=True)
    df.drop(['context','question'],axis=1,inplace=True)
    df.to_csv('~/iclr/pragmatics/global_datasets/task_11_dd.csv',index=False)



if __name__ == "__main__":
    data_path = '~/iclr/pragmatics/data/deixis/dd_2.csv'
    df = read_data(data_path)
    dataset = prepare_dataset(df)
