import random
from gensim.models import Word2Vec
import argparse
import time
import torch.nn as nn
import torch
import numpy as np
import heapq
from tqdm import tqdm

class Node():
    
    def __init__(self, name, features=None, label=None) -> None:
        self.name = name
        self.neighbors = []
        self.features = features
        self.label = label
        self.embeddingVector = None
        self.labelOnehot = None
        self.onehot = None
        
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

def randomWalk(dataset, steps, times):
    walks = []
    freq = {}
    
    for name in dataset:        
        for _ in range(times):
            curWalk = [name]
            nextNode = name
            
            for _ in range(steps):
                node = dataset[nextNode]
                if (not node.neighbors):
                    break
                nextNode = node.neighbors[random.randint(0, len(node.neighbors) - 1)]
                curWalk.append(nextNode)
                
            walks.append(curWalk)
            
    for walk in walks:
        for node in walk:
            if (node in freq):
                freq[node] += 1
            else:
                freq[node] = 1
    a0 = sorted(freq.items(),key=lambda x: (x[1],(x[0],reversed)))
    a1 = sorted(freq.items(),key=lambda x: (x[1],(x[0],reversed)))
    nodeFreq = []
    freqNum = []
    for i in range(len(freq)):
        nodeFreq.append(a0[i][0])
        freqNum.append(a1[i][1])

    return walks, nodeFreq, freqNum

class treeNode():
    
    def __init__(self, lchild=None, rchild=None, embeddingSize=128, name=None) -> None:
        self.vector = torch.randn(
            embeddingSize, 
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.lchild = lchild
        self.rchild = rchild
        self.nodeName = name
        self.path = None

class skipGram(nn.Module):
    
    def __init__(self, wordFreq, freqNum, nodeNum, embeddingSize) -> None:
        super().__init__()

        # skip-gram
        self.embeddingSpace = nn.Linear(nodeNum, embeddingSize)
        
        # hierarchical softMax
        self.path = self.buildTree(wordFreq, freqNum)
        
    def getEmbeddingVector(self, x):
        x = self.embeddingSpace(x)
        return x
        
    def buildTree(self, wordFreq, freqNum):
        heap = []
        nodeList = []
        
        for i in range(len(wordFreq)):
            node = treeNode(name=wordFreq[i])
            node.isNode = False
            nodeList.append((freqNum[i], wordFreq[i], node))
        for node in nodeList:
            heapq.heappush(heap, node)
        
        while (len(heap) != 1):
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)
            weight = node1[0] + node2[0]
            heapq.heappush(
                heap,
                (
                    weight, str(hash(treeNode())),
                    treeNode(node1[2], node2[2])
                )
            )
            
        (_, _, head) = heapq.heappop(heap)
        return self.checkNodePath(head, [], {})
            
    def checkNodePath(self, head, path, pathList):
        # 这个结点是叶子结点(保存的是词信息)
        if (head.nodeName):
            pathList[head.nodeName] = path
            return pathList

        pathList = self.checkNodePath(head.lchild, path + [(head, 'left')], pathList)
        pathList = self.checkNodePath(head.rchild, path + [(head, 'right')], pathList)
        return pathList
    
    def forward(self, x, targetName, device):
        # skip-gram: get embedding vector
        x = self.embeddingSpace(x)
        
        # heirarchical-softmax: count P
        output = torch.tensor(1, device=device)
        path = self.path[targetName]
        for (node, direct) in path:
            Pr = torch.dot(
                x, 
                node.vector 
                if direct == 'left' else torch.neg(node.vector)
            )
            Pr = torch.sigmoid(Pr)
            output = output * Pr
        return output
    
class LossFunc(nn.Module):
    
    def forward(self, x):
        x = torch.log(x)
        x = torch.neg(x)
        return x
            
class Classification(nn.Module):
    
    def __init__(self, embeddingSize, labelSize) -> None:
        super().__init__()
        
        self.layer = nn.Linear(embeddingSize, labelSize)
        
    def forward(self, x):
        x = self.layer(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timeStart = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora', help='now just support "cora" dataset. (random walk)') 
    parser.add_argument('--walk_length', type=int, default=10, help="length of random walk. (random walk)") 
    parser.add_argument('--node_time', type=int, default=5, help="walks per node. (word2vec)")
    parser.add_argument('--embedding_vector_size', type=int, default=128, help='Dimensionality of the word vectors. (word2vec)')
    parser.add_argument('--window_size', type=int, default=2, help='Maximum distance between the current and predicted word within a sentence. (word2vec)')
    parser.add_argument('--mini_count', type=int, default=0, help='Ignores all words with total frequency lower than this. (word2vec)')
    parser.add_argument('--workers', type=int, default=4, help='Use these many worker threads to train the model. (word2vec)')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='The initial learning rate. (word2vec)')
    parser.add_argument('--min_learning_rate', type=float, default=1e-3, help='Learning rate will linearly drop to "learning_rate" as training progresses.')
    parser.add_argument('--word2vec_epochs', type=int, default=10, help='Number of epochs over the corpus. (word2vec)')
    parser.add_argument('--classifi_epochs', type=int, default=100, help='Number of epochs over the corpus. (classification)')
    parser.add_argument('--validation_num', type=float, default=0.2, help='Ratio of validation set. (classification)')
    parser.add_argument('--early_stop', type=bool, default=True, help='Early stop or not. (classification)')  
    parser.add_argument('--min_val_loss', type=float, default=1e-2, help='Minimum change in every validation loss. (classification)') 
    parser.add_argument('--embedding_save_path', type=str, default='save/embedding_vector_deepwalk.npy', help='embedding vectors save path. (save)')
    parser.add_argument('--classification_save_path', type=str, default='save/classifi_deepwalk.pth', help='classification model save path. (save)')   
    args = parser.parse_args()
    
    if (args.dataset == 'cora'):
        data, label = coraUtilities(device)
        dataTime = time.time()
        print('>>> cora: data init success! ({:.2f}s)'.format(dataTime - timeStart))
    walks, nodeFreq, freqNum = randomWalk(data, args.walk_length, args.node_time)
    walkTime = time.time()
    print('>>> randomWalk: generate random walk success! ({:.2f}s)'.format(walkTime - dataTime))
    
    # model = Word2Vec(
    #     sentences=walks,
    #     vector_size=args.embedding_vector_size,
    #     window=args.window_size,
    #     min_count=args.mini_count,
    #     workers=args.workers,
    #     sg=1, hs=1,
    #     alpha=args.learning_rate,
    #     min_alpha=args.min_learning_rate,
    #     epochs=args.word2vec_epochs    
    # )    
    # for word in data:
    #     data[word].embeddingVector = torch.tensor(model.wv[word], device=device)
    skipgram = skipGram(nodeFreq, freqNum, len(data), args.embedding_vector_size)
    skipgram = skipgram.to(device)
    lossFunc = LossFunc()
    lossFunc = lossFunc.to(device)
    optimizer = torch.optim.SGD(skipgram.parameters(), lr=args.learning_rate)
    for epoch in range(args.word2vec_epochs):
        skipgram.train()
        for walk in tqdm(walks):
            for i, word in enumerate(walk):
                input = data[word].onehot
                neighbors = walk[
                    i - args.window_size if i - args.window_size > 0 else 0: i
                    ] + walk[i: i + args.window_size]
                for neighborWord in neighbors:
                    optimizer.zero_grad()
                    output = skipgram(input, neighborWord, device)
                    loss = lossFunc(output)
                    loss.backward()
                    optimizer.step()
    for word in data:
        data[word].embeddingVector = skipgram.getEmbeddingVector(data[word].onehot)
    word2vecTime = time.time()
    print('>>> word2vec: embedding vector trained success! ({:.2f}s)'.format(word2vecTime - walkTime))
    
    classification = Classification(args.embedding_vector_size, len(label))
    classification.to(device)
    lossFunc = nn.CrossEntropyLoss()
    lossFunc.to(device)
    optimizer = torch.optim.Adam(classification.parameters(), lr=1e-2)
    
    # 划分训练集和验证集
    train = data.copy()
    validate = []
    for _ in range(int(len(data) * args.validation_num)):
        key = random.sample(train.keys(), 1)[0]
        validate.append(key)
        del train[key]
    # 训练
    classifiStartTime = time.time()
    lastValiLoss = 0.0
    for epoch in range(args.classifi_epochs):
        curLoss = 0.0
        classification.train()
        for word in train:
            input = data[word].embeddingVector
            target = data[word].labelOnehot
            output = classification(input)
            optimizer.zero_grad()
            loss = lossFunc(output, target)
            loss.backward()
            optimizer.step()
            curLoss += loss
        
        # 每隔30轮训练，进行一次测试
        if (epoch % 30 == 0):
            classification.eval()
            cur = 0
            with torch.no_grad():
                valiLoss = 0.0
                for word in validate:
                    input = data[word].embeddingVector
                    target = data[word].labelOnehot
                    tar = data[word].label
                    output = classification(input)
                    loss = lossFunc(output, target)
                    if (output.argmax() == tar):
                        cur += 1
                    valiLoss += loss
                valiLoss /= len(validate)
                lossChange = lastValiLoss - valiLoss
                if (args.early_stop and abs(lossChange) < args.min_val_loss):
                    classifiEndTime = time.time()
                    print('>>> classification validation: early stop, loss={:.5f}! ({:.2f}s)'.format(valiLoss, classifiEndTime - classifiStartTime))
                    break
                lastValiLoss = valiLoss
            acc = cur / len(validate)
            classifiEndTime = time.time()
            print('>>> classification validation: epoch {}, acc={:.3f}. ({:.2f}s)'.format(epoch, acc, classifiEndTime - classifiStartTime))
        classifiEndTime = time.time()
        print('>>> classification train: epoch {}, loss={:.5f}. ({:.2f}s)'.format(epoch, curLoss / len(train), classifiEndTime - classifiStartTime))
        classifiStartTime = classifiEndTime
    
    # 测试
    cur = 0
    with torch.no_grad():
        for word in data:
            input = data[word].embeddingVector
            target = data[word].label
            output = classification(input)
            if (output.argmax() == target):
                cur += 1
    acc = cur / len(data)
    testTime = time.time()
    print('>>> test: acc = {:.2f}. ({:.2f}s)'.format(acc, testTime - classifiEndTime))        
        
    torch.save(classification, args.classification_save_path)
    array = {}
    for word in data:
        dir = {}
        dir['embeddingVector'] = data[word].embeddingVector
        dir['label'] = data[word].label
        array[word] = dir
    array['label'] = label
    np.save(args.embedding_save_path, array)
    saveTime = time.time()
    print('>>> save: embedding vector save at "{}", classification model save at "{}". ({:.2f}s)'.format(
        args.embedding_save_path, args.classification_save_path, saveTime - testTime
    ))
