#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import time
import json
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import PIL
from PIL import Image
from collections import OrderedDict
import torch
from torch import nn, optim
#import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import os, random

#data path



#Get standred size data for train and test and validate
def preprossessing(train_dir,valid_dir,test_dir):
    train_transforms=transforms.Compose([transforms.RandomRotation(45),
                                     transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
    validate_transforms=transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(244),
                                        transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
    test_transforms=transforms.Compose([transforms.Resize(255),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
    # TODO: Load the datasets with ImageFolder
    image_datasets = None 
    train_data=datasets.ImageFolder(train_dir,transform=train_transforms)
    test_data=datasets.ImageFolder(test_dir,transform=test_transforms)
    valed_data=datasets.ImageFolder(valid_dir,transform=validate_transforms)
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloaders=torch.utils.data.DataLoader(train_data,shuffle=True,batch_size=64)
    testloaders=torch.utils.data.DataLoader(test_data,shuffle=True,batch_size=64)
    valedloaders=torch.utils.data.DataLoader(valed_data,shuffle=True,batch_size=64)
    
    dataloaders=[trainloaders,valedloaders,testloaders]
    imageloaders=[train_data,valed_data,test_data]
    return dataloaders,imageloaders

def modelselection(x):
    if x=='vgg16':
        model=models.vgg16(pretrained=True)
    else:
        model=models.vgg19(pretrained=True)
    for para in model.parameters():
        para.requires_grad=False
    classifier=nn.Sequential(OrderedDict([('fc1',nn.Linear(25088,1024)),
                                     ('drop',nn.Dropout(p=0.2)),
                                     ('relu',nn.ReLU()),
                                     ('fc2',nn.Linear(1024,512)),
                                     ('drop',nn.Dropout(p=0.2)),
                                     ('relu',nn.ReLU()),
                                     ('fc3',nn.Linear(512,102)),
                                     ('output',nn.LogSoftmax(dim=1))]))
    model.classifier=classifier
    
    return model
#model

def Trainnw(model,dataloaders):
    dataloaders=dataloaders
    model=model
    import time
    criterion=nn.NLLLoss()
    lr=float(input('Enter the learing rate: '))
    optimizer=optim.SGD(params=model.classifier.parameters(),lr=lr)

    epoch=int(input('Enter the number of epochs you want: '))
    step=0

    cuda=torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu=input('Enter the you want to run model on GPU Y/N?:')
    if gpu=='Y':
        model.cuda()
    else:
        model.cpu()
    
    running_loss=0
    accuracy=0

    startthecode=time.time()

    for e in range(epoch):
     
        training_nw=0
        valadating_nw=1
    
    
        for i in [0,1]:
             #print(i)
            if i == 0:
            
                model.train()
            else:
            
                model.eval()
            
            step=0
        
            for data in dataloaders[i]:
            
                step+=1
            
                images,lable = data
                
                
                if gpu=='Y':
                
                    images,lable= Variable(images.cuda()), Variable(lable.cuda())
                else:
                    images,lable=Variable(images), Variable(lable)
                
            
                optimizer.zero_grad()
            
                output=model(images)
                loss=criterion(output,lable)
            
                if i == 0:
                
                    loss.backward()
                    optimizer.step()
            
                running_loss+=loss.item()
            
            #Convert lables in terms of prob
            
            
                ps=torch.exp(output)
            
                top_p, top_class = ps.topk(1, dim=1)
            
                equles=top_class==lable.view(*top_class.shape)
            
                accuracy=torch.mean(equles.type(torch.FloatTensor)).item()
            
            if i==0:
                
                print('\nepoch: {}/{}' .format(e+1,epoch))
                print('\nTrainingLoss: {:.4f} '.format(running_loss/step))
                
            else:
                 #print('\nepoch: {}/{}' .format(e+1,epoch))
                print('\nValadatingLoss: {:.4f},\tacc: {:.4f}'.format(running_loss/step,accuracy))
            
            running_loss =0
            
    totaltime=time.time()-startthecode

    print('Total time for code evel: {:.0f}m {:.0f}s'.format(totaltime//60,totaltime%60))
    #model.load_state_dic()
    return model,lr,epoch,optimizer


def checkpoint(x,y,z,p,optimizer):
    
    
    
    checkpoint = {'input_size': 25088,
              'output_size': 102,
              'arch': x,
              'learning_rate': z,
              'batch_size': 64,
              'classifier' : p.classifier,
              'epochs': y,
              'optimizer': optimizer.state_dict(),
              'state_dict': p.state_dict(),
              'class_to_idx': p.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')
    return 'checkpoint.pth'
    

    model=input('Enter which model you wantfrom torchvision models :')

    
def main():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    dataloaders,imageloaders=preprossessing(train_dir,valid_dir,test_dir)
    train_data=imageloaders[0]
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    x=input('Enter which model you want extact from torch models vgg16/19?:')
    model=modelselection(x)
    model,lr,epoch,optimizer=Trainnw(model,dataloaders)
    model.class_to_idx = train_data.class_to_idx
    path=checkpoint(x,epoch,lr,model,optimizer)
    
main()
    
    
    
