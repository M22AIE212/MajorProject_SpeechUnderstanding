import os

import numpy as np
import random
import matplotlib.pyplot as plt
from time import time

import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset

from utils import imshow,show_plot,plotting,Config
from custom_dataset import CustomDataset
from model import SiameseNetwork
from loss import CustomCrossEntropyLoss
from train import train_model
from eval import evaluate_model


print('Version', torch.__version__)
print('CUDA enabled:', torch.cuda.is_available())

if __name__ =="__main__" :

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  learning_rate = 0.001
  counter = []
  loss_history = []
  iteration_number= 0
  EPOCHS = 50

  ## Train and Test Datasets and Dataloaders
  folder_dataset = dataset.ImageFolder(root=Config.training_dir)
  train_dataset = CustomDataset(imageFolderDataset=folder_dataset,
                                          transform=transforms.Compose([#transforms.Resize((227,227)),
                                                                        transforms.ToTensor()
                                                                        ])
                                        ,should_invert=False)
  train_dataloader = DataLoader(train_dataset,
                                shuffle=True,
                                batch_size=Config.train_batch_size)
  
  folder_dataset_test = dataset.ImageFolder(root=Config.testing_dir)
  test_dataset = CustomDataset(imageFolderDataset=folder_dataset_test,
                                          transform=transforms.Compose([#transforms.Resize((100,100)),
                                                                        transforms.ToTensor()
                                                                        ])
                                        ,should_invert=False)

  ## Model , Loss and Optimizer Initialization
  model = SiameseNetwork().to(device)
  criterion = CustomCrossEntropyLoss().to(device)
  optimizer = optim.Adam(model.parameters(), lr = learning_rate)

  ## Training Loop
  train_loss_list = []
  test_loss_list = []
  train_accuracy_list = []
  test_accuracy_list = []

  for current_epoch in range(0, EPOCHS):
      print("Epoch number: ", current_epoch)
      start_time = time()
      
      print("\nTraining:")
      train_loss, train_accuracy = train_model(model, device, train_dataloader, current_epoch, optimizer, criterion)
      
      print("\nTesting:")
      test_loss, test_accuracy = evaluate_model(model, device, test_dataloader, criterion)
      
      train_loss_list.append((current_epoch, train_loss))
      test_loss_list.append((current_epoch, test_loss))
      train_accuracy_list.append((current_epoch, train_accuracy))
      test_accuracy_list.append((current_epoch, test_accuracy))
      
      torch.save(model.state_dict(), 'siamese_net_crossEntropy.pt')
      end_time = time()
      print("Time taken for running epoch {} is {:.3f} seconds.\n\n".format(current_epoch, end_time-start_time))

  test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)
