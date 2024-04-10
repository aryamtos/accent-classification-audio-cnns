import sys
from PIL import Image
sys.path.append("model")
sys.path.append("utilities")
from model.custom_model_ import CustomModel
from model.training import train_model_early_stopping
from dataloader import SpectrogramDataset
from torchvision import transforms
import torch 
import torch.nn as nn
import os
import sys
import argparse
from torch.utils.data import random_split
from torch.utils.data import Subset
from sklearn.model_selection import KFold
from utilities.util import confusion_matrix_plot,plot_heatmap,get_gradcam
from torch.utils.data import DataLoader, ConcatDataset

def gen_heatmap(args):
    checkpoint = args.checkpoint
    batch_size = args.batch_size
    dataset_test = args.dataset_test
    model = CustomModel()
    device = "cuda"if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()
    dataset_test = args.dataset_test
    transform = transforms.Compose([

        transforms.Resize((227,227)),
        transforms.ToTensor()
    ])
    testset = SpectrogramDataset(folder_path=dataset_test, transform=transform)
    img_path = '/store/amatos/pasta/projects/segmentation/src/test_/sp/segment_2_7D7x5o6VPZLmun2LEKe5XR_1_2.png'
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # image, label = testset[1000]
    # denorm_image = image.permute(1,2,0)
    
    # image = image.unsqueeze(0).to(device)
    model.to(device)
    # pred = model(image)
    pred = model(image_tensor)
    heatmap = get_gradcam(model,image_tensor,pred[0][1], size=227)

    denorm_image = image_tensor.squeeze(0).permute(1,2,0)
    # denorm_image = denorm_image * torch.tensor([0.229, 0.224, 0.225]).to(device) + torch.tensor([0.485, 0.456, 0.406]).to(device)
    denorm_image = denorm_image.cpu().numpy()
    plot_heatmap(denorm_image,pred,heatmap)


def evaluate(args):
    checkpoint = args.checkpoint
    batch_size = args.batch_size
    dataset_test = args.dataset_test
    model = CustomModel()
    device = "cuda"if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()
    transform = transforms.Compose([

        transforms.Resize((227,227)),
        transforms.ToTensor()
    ])
    testset = SpectrogramDataset(folder_path=dataset_test, transform=transform)
    testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False)
    y_pred = []
    y_true = []
    confusion_matrix_plot(y_pred,y_true,model,device,testloader)


def train(args):

    dataset_train = args.dataset_train
    dataset_val = args.dataset_val
    batch_size = args.batch_size
    lr = args.lr
    decay = args.decay
    n_epochs = args.n_epochs
    patience = args.patience
    seed = args.seed

    transform = transforms.Compose([

        transforms.Resize((227,227)),
        transforms.ToTensor()
    ])

    trainset = SpectrogramDataset(folder_path=dataset_train, transform=transform)
    validset = SpectrogramDataset(folder_path=dataset_val, transform=transform)


    image,label = trainset[1]
    print(image.shape)

    trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True)
    valloader = torch.utils.data.DataLoader(validset,batch_size=batch_size,shuffle=False)

    for images,labels in trainloader:
        break

    model = CustomModel()
    device = "cuda"if torch.cuda.is_available() else "cpu"

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=lr,eps=1e-8,weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2,eta_min=lr*0.01)
    criterion = torch.nn.CrossEntropyLoss()
    loss = nn.CrossEntropyLoss()
    dataset = ConcatDataset([trainset,validset])
    k_folds=10
    loss_list = []
    acc_list = []
    print(len(dataset))
    results = {}
    torch.manual_seed(seed)
    # cross_validation_(k_folds,n_epochs,loss,dataset,
    #                     optimizer,model,criterion,loss_list,device,results,acc_list)
    model,train,val_loss = train_model_early_stopping(device,trainloader,valloader,model,batch_size,patience,n_epochs,optimizer,scheduler,criterion)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Example of parser")
    subparsers = parser.add_subparsers(dest="mode")

    parser_train = subparsers.add_parser("train")

    parser_train.add_argument('--dataset_train',type=str,required=True,help='Directory of dataset')
    parser_train.add_argument('--dataset_val',type=str,required=True,help='Directory of validation.')
    parser_train.add_argument('--batch_size', type=int,required=True)
    parser_train.add_argument('--lr',type=float,required=True)
    parser_train.add_argument('--decay',type=float,required=True)
    parser_train.add_argument('--n_epochs',type=int,required=True)
    parser_train.add_argument('--patience',type=int,required=True)
    parser_train.add_argument('--seed',type=int,required=True)

    parser_eval = subparsers.add_parser("eval")

    parser_eval.add_argument('--checkpoint',type=str,required=True,help='Directory of checkpoint file')
    parser_eval.add_argument('--batch_size',type=int,required=True)
    parser_eval.add_argument('--dataset_test',type=str,required=True)

    parser_cam = subparsers.add_parser("gradcam")

    parser_cam.add_argument('--checkpoint',type=str,required=True,help='Directory of checkpoint file')
    parser_cam.add_argument('--batch_size',type=int,required=True)
    parser_cam.add_argument('--dataset_test',type=str,required=True)



    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    
    elif args.mode == 'eval':
        evaluate(args)

    elif args.mode == 'gradcam':
        gen_heatmap(args)

    else:
        raise Exception('Error argument!')

