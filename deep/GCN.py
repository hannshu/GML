import argparse
import time
import torch.nn as nn
import torch
from tqdm import tqdm
        
def coraUtilities(device):
    nodeFile = open('Datasets/cora/cora.content')
    edgeFile = open('Datasets/cora/cora.cites')
    
    # init nodes
    nodeIndex = {}
    label = {}
    cnt = 0
    i = 0
    adj = torch.zeros(2708, 2708, device=device)
    X = torch.zeros(2708, 1433, device=device)
    labelMatrix = torch.zeros(2708, 7, device=device)
    for line in nodeFile.readlines():
        line = line.strip('\n').split('\t')
        if (line[-1] not in label):
            label[line[-1]] = cnt
            cnt += 1
        labelMatrix[i][label[line[-1]]] = 1
        features = line[1: -2]
        for j in range(len(features)):
            X[i][j] = int(features[j])
        nodeIndex[line[0]] = i
        i += 1
        
    # build edge
    for line in edgeFile.readlines():
        line = line.strip('\n').split('\t')
        adj[nodeIndex[line[0]]][nodeIndex[line[1]]] = 1
        adj[nodeIndex[line[1]]][nodeIndex[line[0]]] = 1

    adj += torch.eye(len(adj))
    # d = adj.sum(1)
    # d = torch.diag(torch.pow(d , -0.5))
    # adj = d.mm(adj).mm(d)
    
    return adj, X, labelMatrix

class GCN(nn.Module):

    def __init__(self, adj, inFeatures, out) -> None:
        super().__init__()

        self.adj = adj

        self.layer1 = nn.Linear(inFeatures, inFeatures, bias=False)
        self.layer2 = nn.Linear(inFeatures, out, bias=False)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.layer1(self.adj.mm(x)))
        return self.softmax(self.layer2(self.adj.mm(x)))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timeStart = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora', help='now just support "cora" dataset.') 
    parser.add_argument('--learning_rate', type=float, default=1e-2, help="learning rate of GCN") 
    parser.add_argument('--epochs', type=int, default=100, help='training epoch') 
    args = parser.parse_args()
    
    if (args.dataset == 'cora'):
        adj, X, labelMatrix = coraUtilities(device)
        dataTime = time.time()
        print('>>> cora: data init success! ({:.2f}s)'.format(dataTime - timeStart))
    
    gcn = GCN(adj, len(X[0]), len(labelMatrix[0]))
    gcn = gcn.to(device)
    lossFunc = nn.CrossEntropyLoss()
    lossFunc = lossFunc.to(device)
    optimizer = torch.optim.SGD(gcn.parameters(), lr=args.learning_rate)
    
    # train
    gcn.train()
    for epoch in tqdm(range(args.epochs), desc=">>> GCN train"):
        optimizer.zero_grad()
        output = gcn(X)
        loss = lossFunc(output, labelMatrix)
        loss.backward()
        optimizer.step()

    # test
    output = gcn(X)
    pre = output.argmax(dim=1)
    lab = labelMatrix.argmax(dim=1)
    cur = (pre == lab).sum()
    print('>>> test: acc=', int(cur) / len(X))
