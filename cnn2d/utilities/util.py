import torch 
from torch.utils.data import Dataset
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sn
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score,recall_score
from sklearn.metrics import classification_report

def plot_heatmap(denorm_image, pred, heatmap):
    flipped_image = np.fliplr(np.rot90(denorm_image))
    flipped_heatmap = np.fliplr(np.rot90(heatmap))
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(20,20), ncols=3)

    classes = ['mg', 're', 'sp']
    ps = torch.nn.Softmax(dim = 1)(pred).cpu().detach().numpy()
    ax1.imshow(flipped_image)
    ax1.set_xlabel('Tempo')
    ax1.set_ylabel('Frequência')
    ax1.set_title('Imagem Original')

    ax2.barh(classes, ps[0])
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_title('Classe Predita')
    ax2.set_xlim(0, 1.1)

    ax3.imshow(flipped_image)
    ax3.imshow(flipped_heatmap, cmap='magma', alpha=0.7)
    ax3.set_xlabel('Tempo')
    ax3.set_ylabel('Frequência')
    ax3.set_title('Heatmap Sobreposto')
    fig.savefig('segment_2_7D7x5o6VPZLmun2LEKe5XR_1_2.png')
    fig.savefig('segment_2_7D7x5o6VPZLmun2LEKe5XR_1_2.pdf', format='png')
    plt.tight_layout()
    plt.show()


def confusion_matrix_plot(y_pred:list,y_true:list,model,device,testloader):

    model.to(device)
    for inputs,labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)
        output = (torch.max(torch.exp(output),1)[1]).data.cpu().numpy()
        y_pred.extend(output)

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)
    
    classes = ("mg","re","sp")
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                                columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted',fontsize=16)
    plt.ylabel('True',fontsize=16)
    plt.title('Confusion Matrix',fontsize=16)
    plt.tight_layout()
    plt.savefig('output_multiclass.png', dpi=400)
    report = classification_report(y_true, y_pred)
    print(report)
    accuracy = balanced_accuracy_score(y_true, y_pred)
    f1=f1_score(y_true, y_pred, average='weighted')
    print(f1)
    print(accuracy)

def get_gradcam(model,image,label,size):

    label.backward()
    gradients = model.get_activation_gradients()
    #print(gradients)
    pooled_gradients = torch.mean(gradients, dim = [0,2,3])
    #print(pooled_gradients)
    activations = model.get_activation(image).detach()
    #print(activations)

    for i in range(activations.shape[1]):
      activations[:,i,:,:] *= pooled_gradients[i]

    heatmap = torch.mean(activations,dim=1).squeeze().cpu()
    #print(heatmap)
    heatmap = nn.ReLU()(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (size,size))
    #print(heatmap)

    return heatmap