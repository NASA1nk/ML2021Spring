[李宏毅2021春机器学习课程](https://www.bilibili.com/video/BV1Wv411h7kN?p=1)

# 机器学习

Machine Learning ≈ Looking For Function

- **回归Regression**
  - The function outputs a scalar.
- **分类Classification**
  - Given options (classes), the function outputs the correct one.
- **结构化Structured Learning**
  - create something with  structure (image, document)

> 回归：连续
>
> 分类：离散

## 监督学习

### 模型model

带有未知的parameter的function

**Linear models**：
$$
y = b + wx
$$

- x：feature
- w：weight 
- b：bias

### 损失函数Loss

`L(b,w)`

Loss is a function of  parameters

> Define Loss  from Training Data

- MAE（mean absolute error）：平均绝对误差 |$y-\bar{y}$|
- MSE（mean square error）：均方误差 $(y-\bar{y})^2$
- RMSE（Root Mean Squard Error）：均方根误差 $\sqrt{\frac{1}{N}\sum_{n=1}^{N}(f(x^n)-\bar y^n)^2}$

### 最优化Optimization

$$
w^*,b^* = arg\underset{w,b}{min} L
$$

 

> arg min 使L最小的参数w和b

#### 梯度下降Gradient Descent

存在问题：容易陷入局部最优解（local minima）而得不到全局最优解（global minima）

> 负梯度方向，LOSS下降最快

1. (Randomly) Pick an initial value $w^0$

2. computer 
   $$
   \frac{\partial L}{\partial w}|_{w = w^0,b = b^0}\\
   \frac{\partial L}{\partial b}|_{w = w^0,b = b^0}
   $$

3. $$
   w^1 = w^0 - η\ \frac{\partial L}{\partial w}|_{w = w^0,b = b^0}\\
   b^1 = b^0 - η\ \frac{\partial L}{\partial b}|_{w = w^0,b = b^0}
   $$



> η：learning rate
>
> - negative：increase w,b
> - positive：decrease w,b



**Model Bias**

Linear models have severe limitation（表达能力弱）

### Activation Function

#### Sigmoid Function

Piecewise Linear Curves = **constant + sum of a set of piecewise function**

再用Piecewise Linear Curves去逼近出各种continuous function

> piecewise function其实就是Hard sigmoid function
>
> sigmoid function就是一个hyperParameter

$$
y = c \frac{1}{1+e^{-(b+wx1)}}
$$

简写为：
$$
y = c\ sigmoid(b+wx_1)
$$
修改$c,b,w$来逼近各种各样的piecewise function，从而得到各种Piecewise Linear Curves

![sigmoidfunction](MachineLearning.assets/sigmoidfunction.png)


$$
y = b + wx\\
y = b + \sum\limits_i\ c_i\ sigmoid(b_i+w_ix_1)
$$

$$
y = b + \sum\limits_jw_jx_j\\
y = b + \sum\limits_i c_i\ sigmoid(b_i+\sum\limits_j(w_{ij}x_j))
$$

其中：

- i：sigmoid function 个数
- j：feature个数



  取i = j = 3，就有

> wij：weight for xj for i-th sigmoid

$$
r_1 = b_1 + w_{11}x_1 + w_{12}x_2 + w_{13}x_3\\
r_2 = b_2 + w_{21}x_1 + w_{22}x_2 + w_{23}x_3\\
r_3 = b_3 + w_{31}x_1 + w_{32}x_2 + w_{33}x_3\\
$$

改写成向量的形式，就有
$$
\left[\begin{matrix}r_1\\r_2\\r_3\end{matrix}\right] = \left[\begin{matrix}b_1\\b_2\\b_3\end{matrix}\right] +
\left[\begin{matrix}w_{11}&w_{12}&w_{13}\\w_{21}&w_{22}&w_{23}\\w_{31}&w_{32}&w_{33}\end{matrix}\right]
\left[\begin{matrix}x_1\\x_2\\x_3\end{matrix}\right]
$$


简化形式，就有
$$
r = b + Wx​
$$
ri通过sigmoid函数得到ai

![sigmoid过程](MachineLearning.assets/sigmoid过程.png)

所以传入ai后就有
$$
y = b + c^T a
$$
带入ai就有

![sigmoid函数求y](MachineLearning.assets/sigmoid函数求y.png)
$$
y=b+c^T\ σ(b+Wx)
$$
其中：

- feature
  - x
- unknown parameters
  - W
  - cT
  - 第一个b是常量
  - 第二个b是向量

将unknown parameters全部一起拼成新的列向量，也就是Loss Function
$$
θ=\left[\begin{matrix}θ_1\\θ_2\\θ_3\\\vdots\end{matrix}\right]
$$


> 将W矩阵划分成向量再一起拼接

最优化求解
$$
θ^* = arg\underset{θ}{min} L
$$

1. (Randomly) Pick an initial value $θ^0$

2. $$
   g=\left[\begin{matrix}
   \frac{\partial L}{\partial θ_1}|_{θ = θ^0}\\
   \frac{\partial L}{\partial θ_2}|_{θ = θ^0}\\
   \frac{\partial L}{\partial θ_3}|_{θ = θ^0}\\
   \vdots
   \end{matrix}\right]
   $$

   即
   $$
   g=\nabla L(θ^0)
   $$

3. $$
   \left[\begin{matrix}
   θ_1^1\\θ_2^1\\θ_3^1\\\vdots
   \end{matrix}\right]
   =
   \left[\begin{matrix}
   θ_1^0\\θ_2^0\\θ_3^0\\\vdots
   \end{matrix}\right]
   -
   \left[\begin{matrix}
   η\frac{\partial L}{\partial θ_1}|_{θ = θ^0}\\
   η\frac{\partial L}{\partial θ_2}|_{θ = θ^0}\\
   η\frac{\partial L}{\partial θ_3}|_{θ = θ^0}\\
   \vdots
   \end{matrix}\right]
   $$

   即
   $$
   \theta^1=\theta^0-\eta g
   $$

> g：gradient
>
> η：learning rate



#### ReLU

Rectified Linear  Unit
$$
y = c\ max(0,b+wx_1)
$$
则model为
$$
y = b + \sum\limits_{2i} c_i\ max(0,b_i+\sum\limits_j(w_{ij}x_j))
$$


### Batch

将数据集随机分成多个batch

1. 拿出第一个batch的数据来算loss-L1

2. 用L1来算出g
   $$
   g=\nabla L^1(θ^0)
   $$

3. 用g来更新参数
   $$
   \theta^1=\theta^0-\eta g
   $$

4. 再用第二个batch的数据来算loss-L2

5. 用L2来算出g

6. 用g来更新参数

7. ......

**所有的batch计算看成一次，叫做epoch**，每一次的更新参数，叫做Updata

> 做了一次epoch训练是不知道有多少次Updata的，取决于batch size（多少个batch就有多少次Update）
>
> batch size是一个hyperParameter



## 神经元Neuron

Deep =  Many hidder layers

1. 上述的Model被叫做Neuron，多层Neuron被叫做Neuron Network
2. 将每一排的Neuron叫做Hidden Layer，很多Layer就被称作Deep，整个模型就叫Deep Learning

![多个Neuron](MachineLearning.assets/多个Neuron.png)



## 过拟合Overfittting



# PyTorch

An open source **machine learning framework**

A Python package that provides two high-level features

- **Tensor** computation (like NumPy) with strong GPU acceleration
- Deep neural networks built on a tape-based autograd system

> Facebook AI 

![Overview of the DNN Training Procedure](MachineLearning.assets/Overview of the DNN Training Procedure.png)

## Tensor

**张量**

torch中的一种数据结构

High-dimensional matrix (array)

> 各种维度的数组

### Data Type

| Data type               | dtype       | tensor            |
| ----------------------- | ----------- | ----------------- |
| 32-bit floating point   | torch.float | torch.FloatTensor |
| 64-bit integer (signed) | torch.long  | torch.LongTensor  |

### shape

- **1-D tensor**
  - （3，）
- **2-D tensor**
  - （dim0，dim1）
- **3-D tensor**
  - （dim0，dim1，dim2）

> dimension：dim in PyTorch == axis in NumPy
>
> dim0：三维的高度

`x.shape`：返回`torch.Size`

### Constructor

- **From list / NumPy array** 
  - `x = torch.tensor([[1, -1], [-1, 1]])` 
  - `x = torch.from_numpy(np.array([[1, -1], [-1, 1]]))`
- **Specify shape**
  - Zero tensor 
    - `x = torch.zeros([2, 2])` 
  - Unit tensor 
    - `x = torch.ones([1, 2, 5])`

### Operator

- Squeeze：remove the specified dimension with length = 1（降维）
- Unsqueeze：expand a new dimension（升维）
- Transpose：transpose two specified dimensions（转置）
- Cat：concatenate multiple tensors（拼接）

```python
x = torch.zeros([1, 2, 3])
# dim0
x = x.squeeze(0)
# torch.Size([2, 3])
x.shape

x = torch.zeros([2, 3])
# dim1
x = x.unsqueeze(1)
# torch.Size([2, 1, 3])
x.shape

x = torch.zeros([2, 3])
x = x.transpose(0, 1)
# torch.Size([3, 2])
x.shape


x = torch.zeros([2, 1, 3])
y = torch.zeros([2, 3, 3])
z = torch.zeros([2, 2, 3])
# dim1
w = torch.cat([x, y, z], dim=1)
# torch.Size([2, 6, 3])
w.shape


z = x + y
z = x - y
y = x.pow(2)
y = x.sum()
y = x.mean()
```



## Device

Default：tensors & modules will be computed with **CPU**

`model = MyModel().to(device)`

- CPU：`x = x.to('cpu')`
- GPU：`x = x.to('cuda')`

 **GPU**

1. Check if your computer has NVIDIA GPU：`torch.cuda.is_available()`
2. Multiple GPUs：specify 'cuda:0', 'cuda:1'...

> Parallel computing：拆分矩阵运行



## Gradient

计算矩阵微分

`backward()`

1. 
   $$
   x=
   \left[
   \begin{matrix}
   1&0\\-1&1
   \end{matrix}
   \right]
   $$

2. 计算
   $$
   z=\sum_i\sum_j x_{i,j}^2
   $$

3. 计算
   $$
   \frac{\partial z}{\partial x_{i,j}}=2x_{i,j}
   $$

4. 结果
   $$
   \frac{\partial z}{\partial x}=
   \left[
   \begin{matrix}
   2&0\\-2&2
   \end{matrix}
   \right]
   $$

```python
x = torch.tensor([[1., 0.], [-1., 1.]], requires_grad=True)
z = x.pow(2).sum()
# 计算微分
z.backward()
# 查看 tensor([[ 2., 0.],[-2., 2.]])
x.grad
```



## Data

**DataSet**

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    # Read data & preprocess 读数据，预处理
    def __init__(self, file):
      self.data = ...
    # Returns one sample at a time
    def __getitem__(self, index):
      return self.data[index]
    # Returns the size of the dataset
    def __len__(self):
      return len(self.data)
```

**DataLoader**

shuffle

- Training：True
- Testing：False

```python
from torch.utils.data import DataLoader

dataset = MyDataset(file)
tr_set = DataLoader(dataset, batch_size, shuffle=True)
```

![PyTorchData](MachineLearning.assets/PyTorchData.png)



## Neural  Network

`torch.nn`

### Linear Layer

`nn.Linear(in_features, out_features)`

- in_features
- out_features

> in/out_features可以是任意维
>
> *表示任意，但必须要满足矩阵乘法

![PyTorchfeatures](MachineLearning.assets/PyTorchfeatures.png)

![PyTorchlinear](MachineLearning.assets/PyTorchlinear.png)

```python
import torch.nn as nn

layer = torch.nn.Linear(32, 64)
# W:torch.Size([64, 32])
layer.weight.shape
# b:torch.Size([64])
layer.bias.shape
```



### Activation Function

- Sigmoid Activation 
  - `nn.Sigmoid()`
- ReLU Activation 
  - `nn.ReLU()`



### Loss Function

`criterion = nn.MSELoss()`

- Mean Squared Error (for linear regression)
  - `nn.MSELoss()`
- Cross Entropy (for classification)
  - `nn.CrossEntropyLoss()`



### Build

- `nn.Sequential()`
- `forward()`

```python
import torch.nn as nn

class MyModel(nn.Module):
    # Initialize your model & define layers
    def __init__(self):
    	super(MyModel, self).__init__()
		self.net = nn.Sequential(
 			nn.Linear(10, 32),
 			nn.Sigmoid(),
 			nn.Linear(32, 1)
 		)
    # Compute output of your NN
    def forward(self, x):
 		return self.net(x)
```



### Optimizer

`torch.optim`

Stochastic Gradient Descent (SGD)

```python
# params: model.parameters()
# lr: learning rate η
optimizer = torch.optim.SGD(params, lr, momentum = 0)
```



## Training

**准备**

1. Read data via MyDataset
2. Put dataset into Dataloader
3. Contruct model and move to device (cpu/cuda)
4. Set loss function
5. Set optimizer

**训练**

```python
for epoch in range(n_epochs):
    # set model to train mode
    model.train()
    for x, y in tr_set:
        # set gradient to zero
        optimizer.zero_grad()
        # move data to device (cpu/cuda)
        x, y = x.to(device), y.to(device)
        # forward pass (compute output)
        pred = model(x)
        # compute loss
        loss = criterion(pred, y)
        # compute gradient (backpropagation)
        loss.backward()
        # update model with optimizer
        optimizer.step()
```

**Evaluation**

Validation

```python
# set model to evaluation mode
model.eval()
total_loss = 0

for x, y in dv_set:
    x, y = x.to(device), y.to(device)
    # disable gradient calculation
    with torch.no_grad():
        # forward pass (compute output)
        pred = model(x)
        # compute loss
        loss = criterion(pred, y)
        # accumulate loss
        total_loss += loss.cpu().item() * len(x)

# compute averaged loss       
avg_loss = total_loss / len(dv_set.dataset)
```



Testing

```python
# set model to evaluation mode
model.eval()
preds = []
# test dataset
for x in tt_set:
    x = x.to(device)
    with torch.no_grad():
        pred = model(x)
        # collect predictio
        preds.append(pred.cpu())
```



## Save/Load

- **Save**
  - `torch.save(model.state_dict(), path)` 
- **Load** 
  - `ckpt = torch.load(path)` 
  - `model.load_state_dict(ckpt)`

# 深度学习

## 神经网络

[神经元](# 神经元Neuron)

- 把一个Logistic Regression称之为一个Neuron，多个Neuron连接就成为一个Neuron Network
- 每个LR都有自己的wight和bias，所有的LR的weight和bias集合起来就是这个NN的parameter（θ）

### Structure

不同的LR连接就可以得到不同structure的NN

我们定义了NN的structure，就相当于define了一个function set（可以给NN设定不同的参数，它就变成了不同的function）

> Fully Connect Feedforward NetWork：全连接前向传播神经网络
>
> input得到不同的output vector

![FCFFN](MachineLearning.assets/FCFFN.png)

![FCFFN全](MachineLearning.assets/FCFFN全.png)



### Compute

Matrix Operation：转换为矩阵计算，可以使用GPU加速



> y = wx+b

![NN运算](MachineLearning.assets/NN运算.png)

### Feature

output layer的feature不是直接从输入X抽取出来的，而是通过多个hidden layer的计算后抽取出来的一组feature

就相当于自动进行了特征提取（feature extractor）来代替了手动的特征工程（feature engineering）

![NNFeature](MachineLearning.assets/NNFeature.png)





# 神经网络

## 人工神经网络 

ANN：Aritificial Neural Networks

> 简称为神经网络NN：Neural Networks,

**基本思想**

通过大量**简单的神经元**之间的相互连接来**构造复杂的网络结构**，信号（数据）可以在这些神经元之间传递，通过激活不同的神经元和对传递的信号进行**加权**来使得信号被放大或衰减，经过多次的传递来改变信号的强度和表现形式

**MCP模型**

模拟大脑，MCP模型将神经元简化为了三个过程：

- 输入信号**线性加权**
- 求和
- **非线性激活**（阈值法）



## 感知机

Perceptrons

- 输入层（特征向量）
- 隐含层
- 输出层（分类结果）

> 其实就是**两层神经元组成的神经网络**，使用MCP模型对输入的多维数据进行二分类classification

感知机本质上是一种**线性模型**（linear model），**只能处理线性分类问题**，就连最简单的XOR（异或）问题都无法正确分类

## 多层感知机

MLP：Multilayer Perceptrons

有多个隐含层的感知机

可以使用反向传播BP算法，然后使用Sigmoid进行非线性映射，解决非线性分类和学习的问题

### 反向传播BP算法

BP：Backpropagation

**梯度消失问题**

在**误差梯度**后项传递的过程中，后层梯度以乘性方式叠加到前层，由于Sigmoid函数的饱和特性，后层梯度本来就小，误差梯度传到前层时几乎为0，因此无法对前层进行有效的学习

**解决方案**

- 无监督预训练（对权值进行初始化） + 有监督训练微调

- ReLU激活函数能够有效的抑制梯度消失问题（不再需要预训练和微调）

### Sigmoid函数

是一个在生物学中常见的S型的函数，也称为S型生长曲线

在信息科学中，由于其单增以及反函数单增等性质，Sigmoid函数常被用作**神经网络的阈值函数，将变量映射到[0,1]之间**