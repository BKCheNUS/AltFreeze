#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from Blocks import EnsembleModel
from Blocks import CNNBlock1
from Blocks import CNNBlock2
from Blocks import CNNBlock3
from Blocks import MLPBlock
from Training import Trainingfn
#set seed for reproducibility, could do extra for cuda but would slow performance
random.seed(12345)
torch.manual_seed(12345)
np.random.seed(12345)


# In[2]:


torch.cuda.is_available()
torch.cuda.current_device()


# In[3]:


device = torch.device("cuda:0")


# In[4]:


learnrate = 0.001
OPTIM = 'ADAM default'
activation = 'ReLU'
nepochs = 100


# In[5]:


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])
])


# In[6]:


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)


# In[7]:


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[8]:


train_len = len(trainset)
test_len = len(testset)
index = list(range(train_len)) 
print("train:",train_len, "test:", test_len)


# In[9]:


np.random.shuffle(index)
split1 = int(0.45*train_len)
split2 = int(0.9*train_len)
train_index1 = index[:split1]
train_index2 = index[split1:2*split1]
val_index = index[2*split1:]
index2 = list(range(test_len))
np.random.shuffle(index2)
split3 = int(0.1 * test_len)
test_index = index2[:split3]
train_loader1 = torch.utils.data.DataLoader(trainset, sampler = train_index1, batch_size = 50, num_workers = 6)  #batch size 10 because when it was 100 had memory issues
train_loader2 = torch.utils.data.DataLoader(trainset, sampler = train_index2, batch_size = 50, num_workers = 6)  #batch size 10 because when it was 100 had memory issues
val_loader = torch.utils.data.DataLoader(trainset, sampler = val_index, batch_size = 50, num_workers = 6)
test_loader = torch.utils.data.DataLoader(testset, sampler = test_index)  #test set for running every epoch needs to be small
test_loader_big = torch.utils.data.DataLoader(testset)


# In[10]:


print(len(train_loader1))


# In[11]:


trainset1_size = len(train_index1)
trainset2_size = len(train_index1)
val_size = len(val_index)
print("trainset1:",trainset1_size)
print("trainset2:",trainset2_size)
print("val_size:", val_size)


# In[12]:


ensemblemodel = EnsembleModel()
ensemblemodel.to(device)


# In[13]:


optimizer = optim.Adam(ensemblemodel.parameters(), lr = learnrate)
print('optimizer', optimizer)


# In[15]:


criterion = nn.CrossEntropyLoss()


# In[14]:


Trainingfunction1 = Trainingfn(ensemblemodel, train_loader1, criterion, optimizer)
Trainingfunction2 = Trainingfn(ensemblemodel, train_loader2, criterion, optimizer)
trainingloss = []
validationloss = []
testaccuracy = []


# In[ ]:


#print("Time = " time.perf_counter())
for epoch in range(nepochs):
    ensemblemodel.train()
    running_loss = 0.0
    
    
    if epoch%2 == 0 & epoch > 75:
        for param in cnnblock1.parameters():
            param.requires_grad_(False)
        
        for param in cnnblock2.parameters():
            param.requires_grad_(False)
        
        for param in cnnblock3.parameters():
            param.requires_grad_(True)
        
        for param in mlpblock.parameters():
            param.requires_grad_(True)
        trainingval = Trainingfunction1.train()
        trainingloss.append(trainingval)
        print("value:",trainingval)
        #for i, data in enumerate(train_loader1,0):
#			inputs, set2labels = data[0].to(device), data[1].to(device)
	            
#			optimizer.zero_grad()
#	            
#			outputs = ensemblemodel(inputs)
#			loss = criterion(outputs, set2labels)
#			loss.backward()
#			optimizer.step()
#	            #print stats
#			running_loss += loss.item()
#			#Training loss once at the end of each epoch
#			if i%450 == 449:
#				trainingloss.append(running_loss/450)
#				print(running_loss/450)
#				running_loss = 0.0
#			#if i%4500 == 4499:
#				#trainingloss.append(running_loss/4500)
				#print(running_loss/4500)
				#running_loss = 0.0
	
		#Validation loss once at end of epoch
	                
        
    if epoch%2 == 1 & epoch > 75:
        for param in cnnblock1.parameters():
            param.requires_grad_(True)
        
        for param in cnnblock2.parameters():
            param.requires_grad_(True)

        for param in cnnblock3.parameters():
            param.requires_grad_(False)
        
        for param in mlpblock.parameters():
            param.requires_grad_(False)
        trainingval = Trainingfunction2.train()
        trainingloss.append(trainingval)
        print("value:",trainingval)
        
#		for i, data in enumerate(train_loader2,0):
#			inputs, set2labels = data[0].to(device), data[1].to(device)
#	            
#			optimizer.zero_grad()
	            
#			outputs = ensemblemodel(inputs)
#			loss = criterion(outputs, set2labels)
#			loss.backward()
#			optimizer.step()
	            #print stats
#			running_loss += loss.item()
			#Training loss once at the end of each epoch
#			if i%450 == 449:
#				trainingloss.append(running_loss/450)
#				print(running_loss/450)
#				running_loss = 0.0
			#if i%4500 == 4499:
				#trainingloss.append(running_loss/4500)
				#print(running_loss/4500)
				#running_loss = 0.0
	
		#Validation loss once at end of epoch

    if epoch <= 75:
#		for param in cnnblock1.parameters():
#			param.requires_grad_(True)
        
#		for param in cnnblock2.parameters():
#        		param.requires_grad_(True)
        
#		for param in cnnblock3.parameters():
#			param.requires_grad_(True)
        
#		for param in mlpblock.parameters():
#			param.requires_grad_(True)

		
#		for i, data in enumerate(train_loader1,0):
#			inputs, set2labels = data[0].to(device), data[1].to(device)
#	            
#			optimizer.zero_grad()
	            
#			outputs = ensemblemodel(inputs)
#			loss = criterion(outputs, set2labels)
#			loss.backward()
#			optimizer.step()
	            #print stats
#			running_loss += loss.item()
#			#Training loss once at the end of each epoch
#				trainingloss.append(running_loss/450)
#				print(running_loss/450)
#				running_loss = 0.0
			#if i%4500 == 4499:
				#trainingloss.append(running_loss/4500)
				#print(running_loss/4500)
				#running_loss = 0.0
        trainingval = Trainingfunction1.train()
        trainingloss.append(trainingval)
        print("value:",trainingval)

    ensemblemodel.eval()
    running_loss2 = 0.0
    for i,data in enumerate(val_loader): 
        inputs,vallabels = data[0].to(device),data[1].to(device)
        outputs = ensemblemodel(inputs)
        lloss = criterion(outputs, vallabels)   	
                
        running_loss2 += lloss.item()
        if i%len(val_loader) == len(val_loader)-1:
            validationloss.append(running_loss2/len(val_loader))
            print(running_loss2/len(val_loader))
            running_loss2 = 0.0
                        
        #Provides test accuracy at each epoch, 10% of test set     
        # set to testing                           
    correct_count,all_count = 0,0
    for i, data in enumerate(test_loader,0):
        inp,labels = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            logps = ensemblemodel(inp)
        
        ps = torch.exp(logps)
        ps = ps.cpu()
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.cpu()
        if (true_label == pred_label):
            correct_count +=1
        all_count +=1
        
    print("\nModel Accuracy =", (correct_count/all_count))
    testaccuracy.append(correct_count/all_count)
    print(epoch)
	#print("Time = " time.perf_counter())

print("finished training")


# In[ ]:




