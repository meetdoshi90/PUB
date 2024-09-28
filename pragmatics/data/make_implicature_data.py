import pandas as pd
# id,context,question-X,canquestion-X,answer-Y,judgements,goldstandard1,goldstandard2
# Context,Question,Answer,Implicature,Cancellable?,Not sure (0/1),

df1 = pd.read_csv('implicature_merged.csv')
df1.drop(['Not sure (0/1)','Cancellable?','Unnamed: 6'],axis=1,inplace=True)
df1.dropna()
df1.drop_duplicates()
print(df1.columns)
print(len(df1))
df2 = pd.read_csv('circa_implicature_cancellability.csv')
df2.rename(columns={'context': 'Context','question-X':'Question','answer-Y':'Answer'}, inplace=True)

print(df2.columns)

df3 = pd.merge(df1,df2,on=['Context','Question','Answer'],how='inner')
df3.dropna()
df3.drop_duplicates()
print(len(df3))
df3.drop(['id', 'canquestion-X','judgements', 'goldstandard1'],axis=1,inplace=True)
print(df3.columns)
unique_values_A = df3['goldstandard2'].unique()
print(unique_values_A)
print(len(df3))
df_no_nan = df3.dropna(subset=['goldstandard2'])
print(len(df_no_nan))
print(df_no_nan['goldstandard2'].unique())
df_no_nan.to_csv('~/iclr/pragmatics/data/test_imp.csv')
