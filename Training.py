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


class Trainingfn():
	
	def __init__(self, modl, dataset,crit):
		self.modl = modl
		self.dataset = dataset
		self.crit = crit
	def train(self):
		for i, data in enumerate(self.dataset,0):
			inputs, set2labels = data[0].to(device), data[1].to(device)
	            
			optimizer.zero_grad()
	            
			outputs = self.modl(inputs)
			loss = self.crit(outputs, set2labels)
			loss.backward()
			optimizer.step()
	            #print stats
			running_loss += loss.item()
			return(running_loss/len(self.dataset))
		
