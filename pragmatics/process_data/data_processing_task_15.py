import pandas as pd
from datasets import Dataset
from sklearn.utils import shuffle

def read_data(data_path):
    df = pd.read_csv(data_path)
    df = df.dropna()
    return df

# alias,context,options,options_info,label

def prepare_dataset(df):
    # print(len(df))
    df.rename(columns={'label': 'correct answer'}, inplace=True)
    df['pretext'] = df.apply(lambda x: f'Conversation:\n{x["context"]}Question: What does the "{x["alias"]}" refer to?\n', axis=1)
    df.drop(['context','alias'],axis=1,inplace=True)
    print(len(df))
    # df.to_csv('~/iclr/pragmatics/global_datasets/task_15.csv',index=False)



    # options = ['Yes','No','Yes, subject to some conditions','In the middle, neither yes nor no','Other']

    # new_df['options'] = [options] * len(new_df)
    # new_df.rename(columns={'goldstandard2': 'correct answer'}, inplace=True)
    # new_df['pretext'] = new_df.apply(lambda x: f"Context: {x['context']}\nQuestion: {x['question-X']}\nAnswer: {x['answer-Y']}\n", axis=1)
    # new_df.drop(['context','question-X','answer-Y','judgements','goldstandard1','canquestion-X'],axis=1,inplace=True)
    # new_df.to_csv('~/iclr/pragmatics/global_datasets/task_1.csv',index=False)


if __name__ == "__main__":
    data_path = './data/esd_clean-yes-2.csv'
    # prompt_templates_path = './prompt_templates/task_2.csv'
    df = read_data(data_path)
    prepare_dataset(df)