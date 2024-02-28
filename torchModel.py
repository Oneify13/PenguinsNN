import numpy as np
import pandas as pd
import torch
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

torch.manual_seed(50)

data = pd.read_csv("penguins_raw.csv")
data = data.drop("studyName", axis="columns")
data = data.drop("Clutch Completion", axis="columns")
data = data.drop("Island", axis="columns")
data = data.drop("Sample Number", axis="columns")
data = data.drop("Delta 15 N (o/oo)", axis="columns")
data = data.drop("Delta 13 C (o/oo)", axis="columns")
data = data.drop("Date Egg", axis="columns")
data = data.drop("Individual ID", axis="columns")
data = data.drop("Comments", axis="columns")
data = data.drop("Region", axis="columns")
data = data.drop("Stage", axis="columns")

data["Species"] = data["Species"].replace("Adelie Penguin (Pygoscelis adeliae)", 0 )
data["Species"] = data["Species"].replace("Gentoo penguin (Pygoscelis papua)", 1 )
data["Species"] = data["Species"].replace("Chinstrap penguin (Pygoscelis antarctica)", 2)
data["Sex"] = data["Sex"].replace("MALE", 0)
data["Sex"] = data["Sex"].replace("FEMALE", 1)
data = data.dropna(how='any', axis=0)

def min_max_scaling(column):
    return (column - column.min()) / (column.max() - column.min())

for col in data.columns:
    data[col] = min_max_scaling(data[col])

print(data)
class Model(nn.Module):
    # Input layer
    def __init__(self, in_features=5, h1=7, h2=7, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x
    

#Create instance of model
model = Model()

X = data.drop("Species", axis="columns")
y = data["Species"]

# Need to convert to a numpy array 
X = X.values
y = y.values 

# Train, texst, split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

#Convert x features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
#Convert y labels to long tensors 
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
#set the criterion of the model to measure the error, lr = learning_rate 
criterion = nn.CrossEntropyLoss()
#Choose optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Train our model
#How many epochs do we want (sending all our data through the training network once)
epochs = 250
losses = []
print(X_train)
for i in range(epochs):
    y_pred = model.forward(X_train) # Get predicted results
    # Measure Loss
    loss = criterion(y_pred, y_train) # predicted vs actual value
     
    #keep track of our losses
    losses.append(loss.detach().numpy())

    #print every 10 epoch
    if i % 10 == 0 :
        print(f"Epoch: {i} and Loss : {loss}")

    # Do some back propagation (take error rate of forward propagation and feed it back through the network to fine tune the weights)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#Evaluate model on test data set
with torch.no_grad(): # Turn off back propagation
    y_eval = model.forward(X_test) #X_test are features from our test set, 
    loss = criterion(y_eval, y_test) # find loss error

correct = 0 

with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        if y_test[i] == 0:
            x = 'Adelie Penguin (Pygoscelis adeliae)'
        elif y_test[i] == 1:
            x = 'Gentoo penguin (Pygoscelis papua)'
        else:
            x = 'Chinstrap penguin (Pygoscelis antarctica)'
        # Will tell us what type of penguion our network thinks it is 
        print(f'{i + 1}. {str(y_val)} \t {x}')

        # Correct or not 
        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f'Correct: {correct}')


torch.save(model.state_dict(), 'penguinTorch.pt')