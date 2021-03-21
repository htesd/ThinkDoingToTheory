import torch
x=torch.ones(5,5)
x.requires_grad=True
y=x*10
t=y.sum()
print(t)
t.backward(retain_graph=True )
print(x.grad)
x.grad
print(x.grad)
#单纯的执行.grad不会导致梯度的累加
t.backward(retain_graph=True )
#在加了这句话之后计算图就不会被释放，并且每次所求的梯度会不断的累加
print(x.grad)