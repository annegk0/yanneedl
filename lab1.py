from torchvision.datasets import CIFAR10
import numpy as np 
import torchvision.transforms as transforms
import torch 
from torch.utils.data.dataloader import DataLoader

import sys
sys.path.append('efficient-deep-learning/models_cifar100')
import preact_resnet as pr
from sklearn.metrics import accuracy_score

# Parameters 
SAVE_PARAMETERS = True
USE_SUBSET = True

## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network. 
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

### The data from CIFAR10 will be downloaded in the following folder
rootdir = './data/cifar10'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)


#For all the data

trainloader = DataLoader(c10train,batch_size=32,shuffle=True)
testloader = DataLoader(c10test,batch_size=32) 


## number of target samples for the final dataset
num_train_examples = len(c10train)
num_samples_subset = 15000

## We set a seed manually so as to reproduce the results easily
seed  = 2147483647

## Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
indices = list(range(num_train_examples))
np.random.RandomState(seed=seed).shuffle(indices)## modifies the list in place

## We define the Subset using the generated indices 
c10train_subset = torch.utils.data.Subset(c10train,indices[:num_samples_subset])
print(f"Initial CIFAR10 dataset has {len(c10train)} samples")
print(f"Subset of CIFAR10 dataset has {len(c10train_subset)} samples")

# Finally we can define anoter dataloader for the training data
trainloader_subset = DataLoader(c10train_subset,batch_size=32,shuffle=True)
### You can now use either trainloader (full CIFAR10) or trainloader_subset (subset of CIFAR10) to train your networks.

model = pr.PreActResNet18()

n_epochs=2

import torch.optim as optim
import torch.nn as nn

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

best_loss = 100

for epoch in range(n_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader_subset, 0): #les minibatchs sont de taille 32
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if loss<best_loss :
            best_loss = loss
            state = {
                'net': model.state_dict(),
                #'hyperparam': hparam_currentvalue
            }
            torch.save(state, 'bestmodel'+epoch+i+'.pth')

        loss.backward()
        optimizer.step()
        print(i)
        
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

        # print statistics
        accuracy = accuracy_score(labels,outputs)
        print("Accuracy of last model on the subset:  %d", accuracy)
        
    if SAVE_PARAMETERS:

        text = ""
        text += "PARAMETERS :\n\r"
        text += "LOSS_FUNCTION : "+  str(LOSS_FUNCTION) + "\n\r"
        text += "MAX_EXPERIENCE_SIZE : "+  str(MAX_EXPERIENCE_SIZE) + "\n\r"
        text += "EXPERIENCE_MIN_SIZE_FOR_TRAINING : "+  str(EXPERIENCE_MIN_SIZE_FOR_TRAINING) + "\n\r"
        text += "NB_BATCHES : "+  str(NB_BATCHES) + "\n\r"
        text += "BATCH_SIZE : "+  str(BATCH_SIZE) + "\n\r" 
        text += "DISCOUNT_FACTOR : "+  str(DISCOUNT_FACTOR) + "\n\r" 
        text += "EPSILON : "+  str(EPSILON) + "\n\r" 
        text += "LEARNING_RATE : "+  str(LEARNING_RATE) + "\n\r" 
        text += "NB_EPISODES : "+  str(NB_EPISODES) + "\n\r" 
        text += "\n\n"
        text += "Write here the shape of the neural network :\n\n"

        text += "Accuracy of last model on the subset : " + str(accuracy) + "\n\n"
        text += "Loss : " + str(loss) + "\n\n"

        f = open(OUTPUT_DIRECTORY+'/parameters.txt', "w")
        f.write(text)
        f.close()
        

print('Finished Training')