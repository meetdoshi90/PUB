import pandas as pd
import re
from datasets import Dataset
from sklearn.utils import shuffle

def read_data(data_path):
    df = pd.read_csv(data_path)
    df = df.dropna()
    return df
# context,question,answer,explicit_answer,answer_index,options
def prepare_dataset(df):
    # print(len(df))
    df = shuffle(df)
    df = df[:2000]
    # df['options'] = df.apply(lambda row: [row['option1'].split(': ')[1], row['option2'].split(': ')[1], row['option3'].split(': ')[1], row['option4'].split(': ')[1]], axis=1)
    # df['correct answer'] = df.apply(lambda row: row['answer'].split(': ')[1],axis=1)
    df.rename(columns={'explicit_answer': 'correct answer'}, inplace=True)
    df['pretext'] = df.apply(lambda x: f"{x['context']}Speaker_1: {x['question']}\nSpeaker_2: {x['answer']}\n", axis=1)
    # df['pretext'] = df['pretext'].apply(lambda x: re.sub(r'(Correct option= [A-D]:[^\n]+)', r'\1\n', x))
    # df['pretext'] = df['pretext'].apply(lambda x: x.replace('Correct option= ', 'Correct option= Option '))
    df.drop(['context','question','answer','answer_index'],axis=1,inplace=True)
    df.to_csv('~/iclr/pragmatics/global_datasets/task_3.csv',index=False)




    # options = ['Yes','No','Yes, subject to some conditions','In the middle, neither yes nor no','Other']

    # new_df['options'] = [options] * len(new_df)
    # new_df.rename(columns={'goldstandard2': 'correct answer'}, inplace=True)
    # new_df['pretext'] = new_df.apply(lambda x: f"Context: {x['context']}\nQuestion: {x['question-X']}\nAnswer: {x['answer-Y']}\n", axis=1)
    # new_df.drop(['context','question-X','answer-Y','judgements','goldstandard1','canquestion-X'],axis=1,inplace=True)
    # new_df.to_csv('~/iclr/pragmatics/global_datasets/task_1.csv',index=False)


if __name__ == "__main__":
    data_path = './data/grice_implicature_recovery.csv'
    # prompt_templates_path = './prompt_templates/task_2.csv'
    df = read_data(data_path)
    prepare_dataset(df)