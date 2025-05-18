# -*- coding: utf-8 -*-


import my_utils as mu
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

"""## Read Dataset and Create Dataloaders


"""

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # Flip the image horizontally
    transforms.RandomCrop(32, padding=4, padding_mode = 'reflect'), # Crop the image
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Change features of the image
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Shuffle the training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

"""## Create the Model"""

# Block
class Block(nn.Module):
    def __init__(self, num_convs, inputs, outputs):
        super(Block, self).__init__()
        self.num_convs = num_convs
        self.Linear1 = nn.Linear(inputs, num_convs)

        for i in range(num_convs):
            self.add_module(f'conv{i}', nn.Conv2d(inputs, outputs, kernel_size=3, padding=1))
            self.add_module(f'batch{i}', nn.BatchNorm2d(outputs))

        self.silu = nn.SiLU()
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Prepare a layer to match the dimensions for the skip connection, if needed.
        if inputs != outputs:
            self.match_dimensions = nn.Sequential(
                # Use 1x1 kernel to create matching dimensions.
                nn.Conv2d(inputs, outputs, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(outputs)
            )
        else:
            self.match_dimensions = nn.Identity()

    def forward(self, x):
        # Calculate original input
        identity = self.match_dimensions(x)

        pooled = self.pool(x).flatten(1)
        a = self.Linear1(pooled)
        a = self.silu(a)

        last_output = 0
        # Apply convolutions and batch normalizations.
        for i in range((self.num_convs)):
            conv_out = self._modules['conv{0}'.format(i)](x)
            bn_out = self._modules['batch{0}'.format(i)](conv_out)
            bn_out = self.silu(bn_out)  # Apply ReLU after batch normalization.

            scaling_factor = a[:, i].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            scaling_factor = scaling_factor.expand(-1, bn_out.size(1), bn_out.size(2), bn_out.size(3))

            # Scale the output of the conv layer with the i-th coefficient in 'a'
            scaled_conv_out = bn_out * scaling_factor

            # Combine the scaled outputs
            last_output = scaled_conv_out if i == 0 else last_output + scaled_conv_out

        # Apply skip connection
        out = last_output + identity

        return out

# Backbone
class Backbone(nn.Module):
  def __init__(self, conv_arch):
    super(Backbone, self).__init__()
    inputs = 3
    self.conv_arch = conv_arch
    # Create blocks with different number of layers and outputs
    for i, (num_convs, outputs ) in enumerate(conv_arch):
      self.add_module('block{0}'.format(i), Block(num_convs, inputs, outputs))
      inputs = outputs

    # Classifier
    self.last = nn.Sequential(
        nn.AdaptiveAvgPool2d(2),
        nn.Flatten(),
        nn.Linear(outputs * 2 * 2, 1024),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(1024,1024),
        nn.Dropout(p=0.3),
        nn.SiLU(),
        nn.Linear(1024, 10)  # Final Linear layer for classification
    )

  def forward(self, x):
    out = x
    for i in range(len(self.conv_arch)):
      # Calculate each blocks output
      out = self._modules['block{0}'.format(i)](out)
    # Last blocks output goes through classifier
    out = self.last(out)
    return out

def init_weights(m):
    if isinstance(m, nn.Linear):
        # Apply Kaiming Initialization to Linear layers
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        # Apply Kaiming Initialization to Convolutional layers
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

conv_arch = ((1,64), (1,128), (2, 256), (2, 512), (2, 512))
net = Backbone(conv_arch)
net.apply(init_weights)

"""## Create Loss and Optimizer"""

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=0.0002, weight_decay=1e-2)  # Apply the optimal LR


scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,  # Use the learning rate
    final_div_factor=1e3,  # The factor to divide the base_lr by to get the final lr
    steps_per_epoch=len(trainloader),  # Number of batches in one epoch
    epochs=60,  # Total number of epochs
    anneal_strategy='cos'  # Cosine annealing strategy
)

"""## Create Training Script"""

# Training function from Week 9 Lab
def trainf(net, train_iter, test_iter, loss, num_epochs, optimizer, device, scheduler):
    net.to(device)
    animator = mu.d2l.Animator(xlabel='epoch', xlim=[0, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    timer = mu.d2l.Timer()
    for epoch in range(num_epochs):
        metric = mu.d2l.Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            net.train()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                metric.add(l * X.shape[0], mu.d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_loss, train_acc = metric[0] / metric[2], metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter), (train_loss, train_acc, None))
        test_acc = mu.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')

num_epochs = 60
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainf(net, trainloader, testloader, loss, num_epochs, optimizer, device, scheduler)
