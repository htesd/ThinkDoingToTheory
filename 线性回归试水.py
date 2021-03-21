import torch

class Linear(torch.nn.Module):
    def __init__(self,input_dim,output_dim):
        super(Linear,self).__init__()
        self.linear=torch.nn.Linear(input_dim,output_dim)
    def forward(self,x):
        out=self.linear(x)
        return out

x_train=torch.arange(1,100,dtype=torch.float)
y_train=x_train*5.23+0.25
model=Linear(1,1)
epochs=1000
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
criterion=torch.nn.MSELoss()


for e in range(epochs):
    e+=1
    #梯度清零
    optimizer.zero_grad()
    #前向传播
    x=x_train.view(-1,1)
    out=model(x)
    #计算损失
    y=y_train.view(-1,1)
    loss=criterion(out,y)
    #反向
    loss.backward()
    #更新
    optimizer.step()
    if e %50==0:
        print("epoch {},loss{}".format(e,loss.item()))