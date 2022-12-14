import random
from gensim.models import Word2Vec
import argparse
import time
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

class Node():
    
    def __init__(self, name, features=None, label=None, id=0) -> None:
        self.name = name
        self.neighbors = []
        self.features = features
        self.label = label
        self.embeddingVector = None
        self.labelOnehot = None
        self.onehot = None
        self.id = id
        
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
        node = Node(line[0], features=line[1: -2], label=label[line[-1]], id=len(nodeIndex))
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

def randomWalk(dataset, steps, times, p, q):
    prob = [1 / p, 1, 1 / q]
    walks = []
    
    for word in dataset:
        for _ in range(times):
            neighbors = dataset[word].neighbors
            if (not neighbors):
                # 节点本身不存在任何邻居
                walks.append([word])
                break
            curNode = neighbors[random.randint(0, len(neighbors) - 1)]
            walk = [word, curNode]
            preNode = word      # d = 0的结点
            
            for _ in range(steps - 1):
                preNodeNeighbors = dataset[preNode].neighbors
                curNodeNeighbors = dataset[curNode].neighbors
                norNodes = list(set(preNodeNeighbors).intersection(set(curNodeNeighbors)))  # d = 1的结点
                outNodes = list(set(curNodeNeighbors) - set(preNodeNeighbors + [preNode]))  # d = 2的结点
                z = prob[0] + prob[1] * len(norNodes) + prob[2] * len(outNodes)
                curProb = [prob[0] / z, prob[1] / z, prob[2] / z]           # 计算三种情况的概率
                nodes = [preNode] + norNodes + outNodes
                if (not nodes):
                    break
                # nodes中对应节点的概率
                nodeProbs = [curProb[0]] * 1 + [curProb[1]] * len(norNodes) + [curProb[2]] * len(outNodes)
                preNode = curNode                       # 需要记录先前的结点
                curNode = nodes[aliasSample(nodeProbs)]
                walk.append(curNode)
            walks.append(walk)
            
    return walks

def aliasSample(probList) -> int:
    ratio = [ite * len(probList) for ite in probList]
    # accept保存如果取这个格子，则在二项分布时原本是他的概率
    # alias保存如果这个格子中还有别的元素，记录这个元素
    accept, alias = [0] * (len(probList) + 1), [0] * (len(probList) + 1)
    small, large = [], []
    
    # 先将概率分配到small(概率 < 1)和large(概率 > 1)中
    for index, prob in enumerate(ratio):
        if (prob < 1.0):
            small.append(index + 1)
        else:
            large.append(index + 1)
            
    # 如果small和large中都还存在元素，则说明还有large需要和small拼接成1
    while (small and large):
        smallIndex, largeIndex = small.pop(), large.pop()
        # 将small原有的值保存在accept中，将填补的较大概率的元素记录在alias中
        accept[smallIndex] = ratio[smallIndex - 1]
        alias[smallIndex] = largeIndex
        # 将较大概率的那个元素重新装回small/large中
        ratio[largeIndex - 1] -= (1 - ratio[smallIndex - 1])
        if (ratio[largeIndex - 1] < 1.0):
            small.append(largeIndex)
        else:
            large.append(largeIndex)
            
    # 如果还有元素被剩下了，那么这些元素应该概率是1
    for index in large:
        accept[index] = 1
    for index in small:
        accept[index] = 1
    
    return getRandomIndex(accept, alias)

def getRandomIndex(accept, alias):
    listLength = len(accept)
    
    # 先取一个随机数判断取那个格子中的值
    boxIndex = random.randint(1, listLength - 1)
    # 如果这个格子只有这一个元素，则返回这个元素
    if (accept[boxIndex] == 1):
        return boxIndex - 1
    else:
        chooseItem = np.random.uniform(0, 1)
        if (chooseItem <= accept[boxIndex]):
            return boxIndex - 1
        else:
            if (alias[boxIndex] == 0):
                return boxIndex - 1
            else:
                return alias[boxIndex] - 1
            
class skipGram(nn.Module):
    
    def __init__(self, nodeNum, embeddingSize) -> None:
        super().__init__()

        self.embeddingSpace = nn.Linear(nodeNum, embeddingSize)
        self.Wt = nn.Linear(embeddingSize, nodeNum)
        
    def getEmbeddingVector(self, x):
        x = self.embeddingSpace(x)
        return x
    
    def forward(self, x):
        x = self.embeddingSpace(x)
        output = self.Wt(x)
        return output

class LossFunc(nn.Module):
    
    def forward(self, x, positiveItem, negativeItem):
        output = torch.sigmoid(x[positiveItem])
        for item in negativeItem:
            output = output * torch.sigmoid(torch.neg(x[item]))
        output = torch.log(output)
        output = torch.neg(output)
        return output

class Classification(nn.Module):
    
    def __init__(self, embeddingSize, labelSize) -> None:
        super().__init__()
        
        self.layer = nn.Linear(embeddingSize, labelSize)
        
    def forward(self, x):
        x = self.layer(x)
        return x

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    writer = SummaryWriter('logs')
    timeStart = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora', help='now just support "cora" dataset. (random walk)') 
    parser.add_argument('--walk_length', type=int, default=10, help="length of random walk. (random walk)") 
    parser.add_argument('--p', type=int, default=4, help="p score. (random walk)") 
    parser.add_argument('--q', type=int, default=1, help="q score. (random walk)") 
    parser.add_argument('--word2vec_model', default='self-build', help='Use self-build word2vec model or gensim word2vec model. (word2vec)') 
    parser.add_argument('--node_time', type=int, default=5, help="walks per node. (word2vec)")
    parser.add_argument('--embedding_vector_size', type=int, default=128, help='Dimensionality of the word vectors. (word2vec)')
    parser.add_argument('--window_size', type=int, default=2, help='Maximum distance between the current and predicted word within a sentence. (word2vec)')
    parser.add_argument('--mini_count', type=int, default=0, help='Ignores all words with total frequency lower than this. (word2vec)')
    parser.add_argument('--workers', type=int, default=4, help='Use these many worker threads to train the model. (word2vec)')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='The initial learning rate. (word2vec)')
    parser.add_argument('--min_learning_rate', type=float, default=1e-3, help='Learning rate will linearly drop to "learning_rate" as training progresses. (word2vec)')
    parser.add_argument('--word2vec_epochs', type=int, default=10, help='Number of epochs over the corpus. (word2vec)')
    parser.add_argument('--negative_samples', type=int, default=10, help='How many "noise words" should be drawn. (word2vec)')
    parser.add_argument('--classifi_epochs', type=int, default=100, help='Number of epochs over the corpus. (classification)')
    parser.add_argument('--validation_num', type=float, default=0.2, help='Ratio of validation set. (classification)')
    parser.add_argument('--early_stop', type=bool, default=True, help='Early stop or not. (classification)')  
    parser.add_argument('--min_val_loss', type=float, default=1e-2, help='Minimum change in every validation loss. (classification)') 
    parser.add_argument('--embedding_save_path', type=str, default='save/embedding_vector_word2vec.npy', help='embedding vectors save path. (save)')
    parser.add_argument('--classification_save_path', type=str, default='save/classifi_word2vec.pth', help='classification model save path. (save)')   
    args = parser.parse_args()
    
    if (args.dataset == 'cora'):
        data, label = coraUtilities(device)
        dataTime = time.time()
        print('>>> cora: data init success! ({:.2f}s)'.format(dataTime - timeStart))
    walks = randomWalk(data, args.walk_length, args.node_time, args.p, args.q)
    walkTime = time.time()
    print('>>> randomWalk: generate random walk success! ({:.2f}s)'.format(walkTime - dataTime))
    
    if (args.word2vec_model == 'self-build'):
        print('>>> using self-build word2vec model!')
        skipgram = skipGram(len(data), args.embedding_vector_size)
        skipgram = skipgram.to(device)
        word2vecLossFunc = LossFunc()
        word2vecLossFunc = word2vecLossFunc.to(device)
        word2vecOptimizer = torch.optim.SGD(skipgram.parameters(), lr=args.learning_rate)

        # cnt = 0
        for epoch in range(args.word2vec_epochs):
            curLoss = 0.0
            skipgram.train()
            for walk in tqdm(walks, 'word2vec epoch ' + str(epoch)):
                for i, word in enumerate(walk):
                    input = data[word].onehot
                    neighbors = walk[
                        i - args.window_size if i - args.window_size > 0 else 0: i
                        ] + walk[i: i + args.window_size]
                    for neighborWord in neighbors:
                        negetiveList = [
                            random.randint(0, len(data) - 1)
                            for _ in range(args.negative_samples)
                        ]
                        word2vecOptimizer.zero_grad()
                        output = skipgram(input)
                        loss = word2vecLossFunc(output, data[neighborWord].id, negetiveList)
                        loss.backward()
                        word2vecOptimizer.step()
            #             curLoss += loss
            #             cnt += 1
            # writer.add_scalar('word2vec train', curLoss / cnt, epoch)
        for word in data:
            data[word].embeddingVector = skipgram.getEmbeddingVector(data[word].onehot)
    else:  
        print('>>> Using gensim.word2vec model!')
        model = Word2Vec(
            sentences=walks,
            vector_size=args.embedding_vector_size,
            window=args.window_size,
            min_count=args.mini_count,
            workers=args.workers,
            sg=1, hs=0,
            negative=args.negative_samples,
            alpha=args.learning_rate,
            min_alpha=args.min_learning_rate,
            epochs=args.word2vec_epochs    
        )    
        for word in data:
            data[word].embeddingVector = torch.tensor(model.wv[word], device=device)
    word2vecTime = time.time()
    print('>>> word2vec: embedding vector trained success! ({:.2f}s)'.format(word2vecTime - walkTime))
    
    classification = Classification(args.embedding_vector_size, len(label))
    classification.to(device)
    classifiLossFunc = nn.CrossEntropyLoss()
    classifiLossFunc.to(device)
    classifiOptimizer = torch.optim.Adam(classification.parameters(), lr=1e-2)
    
    train = data.copy()
    validate = []
    for _ in range(int(len(data) * args.validation_num)):
        key = random.sample(train.keys(), 1)[0]
        validate.append(key)
        del train[key]
    classifiStartTime = time.time()
    lastValiLoss = 0.0
    for epoch in tqdm(range(args.classifi_epochs), desc="classification: "):
        curLoss = 0.0
        classification.train()
        for word in train:
            input = data[word].embeddingVector
            target = data[word].labelOnehot
            output = classification(input)
            classifiOptimizer.zero_grad()
            loss = classifiLossFunc(output, target)
            loss.backward(retain_graph=True)
            classifiOptimizer.step()
            curLoss += loss
        # writer.add_scalar('classification train', curLoss / len(train), epoch)
        
        classification.eval()
        cur = 0
        with torch.no_grad():
            valiLoss = 0.0
            for word in validate:
                input = data[word].embeddingVector
                target = data[word].labelOnehot
                tar = data[word].label
                output = classification(input)
                loss = classifiLossFunc(output, target)
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
        # writer.add_scalar('classification validation', acc, epoch / 30)
    classifiEndTime = time.time()
    print('>>> classification train: classification model trained success! loss={:.5f}. ({:.2f}s)'.format(curLoss / len(train), classifiEndTime - classifiStartTime))
    
    # 测试
    cur = 0
    testStartTime = time.time()
    with torch.no_grad():
        for word in data:
            input = data[word].embeddingVector
            target = data[word].label
            output = classification(input)
            if (output.argmax() == target):
                cur += 1
    acc = cur / len(data)
    testEndTime = time.time()
    print('>>> test: acc = {:.3f}. ({:.2f}s)'.format(acc, testEndTime - testStartTime))        
        
    # torch.save(classification, args.classification_save_path)
    # array = {}
    # for word in data:
    #     dir = {}
    #     dir['embeddingVector'] = data[word].embeddingVector
    #     dir['label'] = data[word].label
    #     array[word] = dir
    # array['label'] = label
    # np.save(args.embedding_save_path, array)
    # saveTime = time.time()
    # print('>>> save: embedding vector save at "{}", classification model save at "{}". ({:.2f}s)'.format(
    #     args.embedding_save_path, args.classification_save_path, saveTime - testTime
    # ))
