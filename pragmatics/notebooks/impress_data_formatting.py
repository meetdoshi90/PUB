import pandas as pd
import json
import os
import random

folder_path = '../data/IMPPRES/implicature/'
all_data = []

for file in os.listdir(folder_path):
    if file.endswith('.jsonl'):
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            random_lines = random.sample(lines, 300)
            for line in random_lines:
                data = json.loads(line)
                all_data.append(data)

df = pd.DataFrame(all_data)
# df.drop(['trigger1','trigger2','control_item','UID','pairID','paradigmID','trigger','presupposition'],axis=1,inplace=True)
print(len(df))
df.to_csv('~/iclr/pragmatics/data/imppres_implicature.csv',index=False)