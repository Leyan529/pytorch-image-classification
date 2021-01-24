import torch
from read_data import MyData

torch.manual_seed(1)    
BATCH_SIZE = 5
LEARNING_RATE = 0.01


root_dir = "DataSet/MongoLeaf/train"
A_dataset = MyDataLabel(root_dir,'healthy')
B_dataset = MyDataLabel(root_dir,'diseased')
train_dataset = A_dataset + B_dataset

loader = Data.DataLoader(
    dataset=torch_dataset,      # TensorDataset类型数据集
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 设置随机洗牌
    num_workers=2,              # 加载数据的进程个数
)

'''
如果支持GPU
'''
if torch.cuda.is_available():
    linear = linear.cuda()      # 将网络中的参数和缓存移到GPU显存中

'''
训练
'''
for epoch in range(3):   # 训练3轮
    for step, (x, y) in enumerate(loader):  # 每一步
        
        # 如果支持GPU,将训练数据移植GPU显存
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        
        output = linear(x)                  # 预测一下
        loss = loss_func(output, b_y)       # 计算损失
        print('优化了一步之后, 现在的损失是: ', loss.data[0])
        optimizer.zero_grad()               # 清空上一步的梯度缓存
        loss.backward()                     # 计算新梯度, 反向传播
        optimizer.step()                    # 优化一步        
        
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ', batch_x.numpy(), '| batch y: ', batch_y.numpy())
