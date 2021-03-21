import torch
x1=torch.empty(5,3)
x2=torch.range(1,3)
print(x2)
'''
均匀分布：torch.rand()

标准正态分布：torch.randn()

离散正态分布：torch.normal()

线性间距向量：torch.linespace()
'''
x3=torch.zeros(5,3)
x4=torch.tensor([1,2])
x1.size()
x5=torch.add(x1,x2)
x5=x5.view(15)
x6=x5.numpy()
x7=torch.from_numpy(x6)


