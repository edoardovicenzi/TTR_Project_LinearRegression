import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

D = torch.tensor(pd.read_csv("Test\linreg-multi-synthetic-2.csv", header=None).values, dtype=torch.float)

x_dataset = D[:,0:2].t()
y_dataset = D[:, 2].t()

n = 2

A = torch.randn((1,n), requires_grad=True)
b = torch.randn((1), requires_grad=True)

def model(x_input):
    return A.mm(x_input) + b

def loss(y_predicted, y_target):
    return ((y_predicted - y_target)**2).sum()

optimizer = optim.Adam([A,b], lr=0.1)

for t in range(2000):
    optimizer.zero_grad()
    y_predicted = model(x_dataset)
    current_loss = loss(y_predicted, y_dataset)
    current_loss.backward()
    optimizer.step()
    print(f"t = {t}, loss = {current_loss}, A = {A.detach().numpy()}, b = {b.item()}")