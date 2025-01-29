#THIS CODE IS ONLY FOR THE CLASSIFICATION MODELS!

import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#choose the predicted labels csv file.
df = pd.read_csv('predicted_labels_nocutoff.csv')

actual = df['label']
prediction = df['prediction_0']

cm = confusion_matrix(actual, prediction)
tn, fp, fn, tp = cm.ravel()

pos_neg = np.array([['TP: {}'.format(tn), 'FN: {}'.format(fp)],
                   ['FP: {}'.format(fn), 'TN: {}'.format(tp)]])

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=pos_neg, fmt='', cmap='Blues', cbar=False, xticklabels=['High', 'Low'], yticklabels=['High', 'Low'])
plt.xlabel('Predicted')
plt.ylabel('Ground Truth')
plt.title('Confusion Matrix')
plt.show()
