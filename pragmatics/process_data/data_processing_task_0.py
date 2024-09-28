import pandas as pd
from datasets import Dataset
from sklearn.utils import shuffle
import csv

def read_data(circa_data_path):
    df = pd.read_csv(circa_data_path, sep='\t')
    df = df.dropna()
    # df = df[:200]

    # prompt = pd.read_csv(prompt_templates_path, sep=';')
    return df

def prepare_dataset(df):
    new_context = df['context'].tolist() * 2
    new_question = df['question-X'].tolist() * 2
    new_answer = df['canquestion-X'].tolist() + df['answer-Y'].tolist()
    new_df = pd.DataFrame({'context': new_context, 'question': new_question, 'answer': new_answer})
    new_df['type'] = ['Direct answer'] * len(df) + ['Indirect answer'] * len(df)
    new_df = shuffle(new_df)
    new_df = new_df.groupby('context').head(250)
    # label_counts = new_df['type'].value_counts()

    # print(label_counts,len(new_df))
    options = ['Direct answer','Indirect answer']
    # # from sklearn.utils import shuffle
    # # new_df = shuffle(new_df)
    # # random_examples = new_df.sample(4)
    # # print(random_examples)

    # # prompt_template_k_shot = prompt.loc[prompt['type'] == 'k-shot', 'prompt_template'].values[0]
    # # prompt_template_k_shot = prompt_template_k_shot.replace('\\n', '\n')
    new_df['options'] = [options] * len(new_df)
    new_df.rename(columns={'type': 'correct answer'}, inplace=True)
    new_df['pretext'] = new_df.apply(lambda x: f"Context: {x['context']}\nQuestion: {x['question']}\nResponse: {x['answer']}\n", axis=1)
    new_df.drop(['context','question','answer'],axis=1,inplace=True)
    new_df.to_csv('~/iclr/pragmatics/global_datasets/task_0.csv')

if __name__ == "__main__":
    circa_data_path = '../data/circa-data.tsv'
    # prompt_templates_path = './prompt_templates/task_1.csv'
    df = read_data(circa_data_path)
    prepare_dataset(df)
