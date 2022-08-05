# 仓库结构
``` shell
.  
├── Datasets    # 常用图数据集
│   ├── cora    # # cora论文数据集
│   └── ppi     # # ppi蛋白质数据集
├── README.md   # 仓库目录
├── demo.ipynb  # 示例代码
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

## shallow embedding:

### deepwalk
[源码](shallow/deepwalk.py) [论文](http://doi.acm.org/10.1145/2623330.2623732)  



### node2vec


### LINE



## deep embedding(GNN)

### GCN


### GraphSAGE


### GAT


### GIN
