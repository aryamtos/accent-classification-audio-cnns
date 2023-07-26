import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch


classes=('RE','SP')

class ConfusionMatrixVisualizer():

    def __init__(self,model,traindataloader,classes=classes):
        self.model = model
        self.traindataloader = traindataloader
        self.classes = classes
    def predict(self):
        y_pred=[]
        y_true=[]

        for inputs,labels in self.traindataloader.dev:
            output=self.model(inputs)
            output=(torch.max(torch.exp(output),1)[1]).data.cpu().numpy()
            y_pred.extend(output)

            labels=labels.data.cpu().numpy()
            y_true.extend(labels)

        return y_true,y_pred

    def plot_confusion_matrix(self):

        y_true,y_pred = self.predict()
        cf_matrix = confusion_matrix(y_true,y_pred)
        row_sums = np.sum(cf_matrix,axis=1)
        valid_row_sums = np.where(row_sums!= 0, row_sums, 1)
        normalized_matrix = cf_matrix / valid_row_sums[:, None]
        df_cm = pd.DataFrame(normalized_matrix, index=[i for i in self.classes], columns=[i for i in self.classes])

        plt.figure(figsize=(8, 6))
        sn.heatmap(df_cm, annot=True, cmap='Blues')
        plt.xlabel('Predicted', fontsize=16)
        plt.ylabel('True', fontsize=16)
        plt.title('Confusion Matrix', fontsize=16)
        plt.tight_layout()
        plt.savefig('pe_mg_cross_dataset.png', dpi=400)
        plt.show()





