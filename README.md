# 仓库结构
``` shell
.  
├── Datasets    # 常用图数据集
│   ├── cora    # # cora论文数据集
│   └── ppi     # # ppi蛋白质数据集
├── README.md   # 仓库目录
├── deep        # 深度方法(GNN)
│   ├── GCN
│   ├── GraphSAGE
│   ├── GAT
│   └── GIN
└── shallow     # shallow embedding
    ├── deepwalk
    ├── node2vec
    └── LINE
```

# 数据集介绍

## cora
[来源](https://linqs.org/datasets/#cora)  
Cora数据集包含2708篇科学出版物， 5429条边，总共7种类别。
数据集中的每个出版物都由一个 0/1 值的词向量描述，表示字典中相应词的缺失/存在。
该词典由 1433 个独特的词组成。意思就是说每一个出版物都由1433个特征构成，
每个特征仅由0/1表示。

## ppi(经过预处理)
[来源](http://snap.stanford.edu/graphsage/ppi.zip)  
PPI数据集共24张图，每张图对应不同的人体组织，平均每张图有2371个节点，
共56944个节点818716条边，每个节点特征长度为50，其中包含位置基因集，
基序集和免疫学特征。基因本体基作为label(总共121个)，label不是one-hot编码。

# 模型介绍
源码基本均采用pytorch实现  
一些模型中附有库模型作为参考  

## shallow embedding:

### deepwalk
[源码](shallow/deepwalk.py) [论文](http://doi.acm.org/10.1145/2623330.2623732)  
Deepwalk是一种将随机游走和skip-gram两种方法相结合的图结构数据挖掘算法。   

- 随机游走  
对于图中的每个节点，对节点进行随机游走，生成节点序列，用来给下面的skip-gram模型进行训练。  

- skip-gram  
[skip-gram参考资料](https://arxiv.org/abs/1411.2738v4)  

### node2vec
[源码](shallow/node2vec.py) [论文](https://dl.acm.org/doi/10.1145/2939672.2939754)  
与deepwalk类似，node2vec同样采用skip-gram模型进行embedding的训练
但不同的是，node2vec采用negative sampling的方式来优化模型。  

- 随机游走  
这里的随机游走加入了两个超参数p和q来控制向前探索的趋势。  
返回概率p: p比较大的时候，更倾向于向更远的地方进行游走。    
出入概率q: q比较大的时候，更倾向于在**起点的邻居结点**之间游走。  

- alias sampling  
node2vec采用别名采样的方法，把一个结点的按概率的随机采样转化为一个均匀采样和一个01采样。  
[详细解释](https://zhuanlan.zhihu.com/p/111885669)  

### LINE



## deep embedding(GNN)

### GCN


### GraphSAGE


### GAT


### GIN
