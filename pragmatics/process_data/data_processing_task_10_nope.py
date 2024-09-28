import pandas as pd
from datasets import Dataset
from sklearn.utils import shuffle

def read_data(data_path):
    df = pd.read_json(data_path,lines=True)
    print(df.label.unique())
    #df = df.dropna()
    # print(df.head())
    return df


def prepare_dataset(df):
    print(len(df))

    new_options = {
    'E': 'Hypothesis is definitely true given premise',
    'N': 'Hypothesis might be true given premise',
    'C': 'Hypothesis is definitely not true given premise'
    }
    options = ['Hypothesis is definitely true given premise','Hypothesis might be true given premise','Hypothesis is definitely not true given premise']
    
    df['options'] = [options] * len(df)
    # df.rename(columns={'label': 'correct answer'}, inplace=True)
    df['correct answer'] = df['label'].replace(new_options)
    df['pretext'] = df.apply(lambda x: f"Premise: {x['premise']}\nHypothesis: {x['hypothesis']}", axis=1)
    df.drop(['uid','premise','hypothesis','label','metadata'],axis=1,inplace=True)
    df.to_csv('~/iclr/pragmatics/global_datasets/task_10_nope.csv',index=False)


if __name__ == "__main__":
    data_path = './data/nope.jsonl'
    # prompt_templates_path = './prompt_templates/task_2.csv'
    df = read_data(data_path)
    prepare_dataset(df)