import pandas as pd
from datasets import Dataset
from sklearn.utils import shuffle

def read_data(data_path):
    df = pd.read_csv(data_path)
    #df = df.dropna()
    print(df.head())
    return df


def prepare_dataset(df):
    print(len(df))

    new_options = {
    'entailment': 'Hypothesis is definitely true given premise',
    'neutral': 'Hypothesis might be true given premise',
    'contradiction': 'Hypothesis is definitely not true given premise'
    }
    options = ['Hypothesis is definitely true given premise','Hypothesis might be true given premise','Hypothesis is definitely not true given premise']
    
    df['options'] = [options] * len(df)
    df.rename(columns={'gold_label': 'correct answer'}, inplace=True)
    df['correct answer'] = df['correct answer'].replace(new_options)
    df['pretext'] = df.apply(lambda x: f"Premise: {x['sentence1']}\nHypothesis: {x['sentence2']}", axis=1)
    df.drop(['sentence1','sentence2'],axis=1,inplace=True)
    df.to_csv('~/iclr/pragmatics/global_datasets/task_10.csv',index=False)


if __name__ == "__main__":
    data_path = './data/imppres_presupposition.csv'
    # prompt_templates_path = './prompt_templates/task_2.csv'
    df = read_data(data_path)
    prepare_dataset(df)