import random
import argparse
import time
import torch.nn as nn
import torch
import numpy as np
import heapq
from tqdm import tqdm

class Node():
    
    def __init__(self, name, features=None, label=None, weight=1) -> None:
        self.name = name
        self.neighbors = []
        self.features = features
        self.label = label
        self.embeddingVector = None
        self.labelOnehot = None
        self.onehot = None
        self.weight = weight
        
def coraUtilities(device):
    nodeFile = open('Datasets/cora/cora.content')
    edgeFile = open('Datasets/cora/cora.cites')
    
    # init nodes
    nodeIndex = {}
    label = {}
    cnt = 0
    for line in nodeFile.readlines():
        line = line.strip('\n').split('\t')
        if (line[-1] not in label):
            label[line[-1]] = cnt
            cnt += 1
        node = Node(line[0], features=line[1: -2], label=label[line[-1]])
        nodeIndex[node.name] = node
        
    # build edge
    for line in edgeFile.readlines():
        line = line.strip('\n').split('\t')
        if (line[1] not in nodeIndex[line[0]].neighbors):
            nodeIndex[line[0]].neighbors.append(line[1])
        if (line[0] not in nodeIndex[line[1]].neighbors):
            nodeIndex[line[1]].neighbors.append(line[0])
        
    for word in nodeIndex:
        labelVector = torch.zeros(len(label), 
            dtype=torch.float32, device=device)
        labelVector[nodeIndex[word].label] = 1
        nodeIndex[word].labelOnehot = labelVector
        
    for i, word in enumerate(nodeIndex):
        onehot = torch.zeros(len(nodeIndex), device=device)
        onehot[i] = 1
        nodeIndex[word].onehot = onehot
    
    return nodeIndex, label

class LINE(nn.Module):
    
    def __init__(self, dataSize, embeddingSize=128, direct=False) -> None:
        super().__init__()
        
        # 1-order
        self.embeddingSpace1 = nn.Linear(dataSize, int(embeddingSize / 2))
        
        # 2-order
        self.embeddingSpace2 = nn.Linear(dataSize, int((embeddingSize + embeddingSize % 2) / 2))
        self.contextVector = nn.Linear(int((embeddingSize + embeddingSize % 2) / 2), dataSize)
        
        self.direct = direct
        
    def forward(self, x):
        order1 = self.embeddingSpace1(x)
        
        order2 = self.embeddingSpace2(x)
        order2 = self.contextVector(order2) 
        
        return order1, order2
    
class LINELoss(nn.Module):
    
    def forward(self, x):
        pass