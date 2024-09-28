import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv('/raid/nlp/pranavg/iclr/Results/generated_texts/task_1/llama-2-70b-chat/few_shot_k3_run0_mcqa.csv')
df_confusion = pd.crosstab(df.correct_options, df.generate_predictions)
# print(df_confusion)
# cm = confusion_matrix(df.correct_options, df.generate_predictions)
# display = ConfusionMatrixDisplay(cm).plot()
print(df.correct_options.unique())
cm = confusion_matrix(df.correct_options, df.generate_predictions)
ConfusionMatrixDisplay(cm, display_labels=['A' 'B' 'C' 'D' 'E']).plot()
plt.show()