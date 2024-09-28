import pandas as pd
from datasets import Dataset
from sklearn.utils import shuffle

def read_data(circa_data_path):
    df = pd.read_csv(circa_data_path)
    df = df.dropna()
    return df
# ,Context,Question,Answer,Implicature,goldstandard2
def prepare_dataset(df):

    # new_df = df.groupby['goldstandard2'].head(2000)
    new_df = df
    # print(label_counts)
    options = ['Yes','No','Yes, subject to some conditions','In the middle, neither yes nor no','Other']

    new_df['options'] = [options] * len(new_df)
    new_df.rename(columns={'goldstandard2': 'correct answer'}, inplace=True)
    new_df['pretext'] = new_df.apply(lambda x: f"Context: {x['Context']}\nX: {x['Question']}\nY: {x['Answer']}\nImplied meaning: {x['Implicature']}\n", axis=1)
    new_df.drop(['Context','Question','Answer','Implicature'],axis=1,inplace=True)
    new_df.to_csv('~/iclr/pragmatics/global_datasets/task_2.csv',index=False)


if __name__ == "__main__":
    circa_data_path = './data/test_imp.csv'
    # prompt_templates_path = './prompt_templates/task_2.csv'
    df = read_data(circa_data_path)
    prepare_dataset(df)