import pandas as pd
from datasets import Dataset
from sklearn.utils import shuffle

def read_data(circa_data_path):
    df = pd.read_csv(circa_data_path)
    df = df.dropna()
    return df
# dialog_id,Conversation,Presupposition,Tag
def prepare_dataset(df):

    # new_df = df.groupby['goldstandard2'].head(2000)
    new_df = df
    # print(label_counts)
    options = ['Valid','Invalid']

    new_df['options'] = [options] * len(new_df)
    new_df.rename(columns={'Tag': 'correct answer'}, inplace=True)
    new_df['pretext'] = new_df.apply(lambda x: f"Conversation:\n{x['Conversation']}\nAssumption: {x['Presupposition']}\n", axis=1)
    new_df.drop(['dialog_id','Conversation','Presupposition'],axis=1,inplace=True)
    new_df.to_csv('~/iclr/pragmatics/global_datasets/task_13.csv',index=False)


if __name__ == "__main__":
    circa_data_path = './data/presupp_formatted.csv'
    # prompt_templates_path = './prompt_templates/task_2.csv'
    df = read_data(circa_data_path)
    prepare_dataset(df)