[TOC]

## Assignment 1 Note

### 概念

hyperparameter
hyperparameter tuning
training set
validation set
test set：测试集不能过早引入，否则会造成对于测试集的过拟合。
cross-validation：用在小数据集上，但深度学习中不常用。

### Construct a Neural Network

three component:
1. score function(neural network)
2. loss function
3. optimization

### Loss function

$$
L = \underbrace{{1 \over N} \sum_i^{pic\ num} L_i}_{data\ loss} + \underbrace{\lambda \sum_j^{layer\ num} R_j}_{regularization\ loss}
$$
regularization：控制模型的复杂程度，防止过拟合

#### SVM - hinge loss

$$
L_i = \sum_{j \neq y_i} max(0, s_j - s_{y_i} + \Delta), 通常\Delta = 1
$$

#### Softmax - cross-entropy loss

$$
L_i = -log({e^{s_{y_i}} \over {\sum_j e^{s_j}}})
$$
或者
$$
L_i = -s_{y_i} + log \sum_j e^{s_j}
$$
实际使用中，为了防止上溢和下溢，通常会让`s -= max(s)`，这样不影响softmax函数的值，是因为
$$
{e^{s_{y_i}} \over \sum_j e^{s_j}} = {C e^{s_{y_i}} \over C \sum_j e^{s_j}} = {e^{s_{y_i}+\log C} \over \sum_j e^{s_j+\log C}},\ \ \log C = -max_j s_j
$$

### Optimization

调整参数以最小化损失函数。凸函数优化问题。

#### Gradient Descent

损失函数L是一个以所有参数构成的向量作为自变量的多元函数，其梯度是一个与参数向量相同大小的向量。（根据多元微积分）L在某个方向上的方向导数是梯度与该方向上的单位向量的点积，又由点积的性质可知，方向导数最小的方向一定是与梯度向量相反的方向，即下降最快的方向是负梯度方向。

梯度下降法指每次求出梯度，然后将自变量减去梯度与步长的乘积。

学习率（learning rate）即上式中的步长，是最重要的超参数之一。

#### Stochastic Gradient Descent (SGD)

如果每次都使用训练集的所有数据计算梯度，所需要的计算资源太大，因而每次随机选出一部分数据，称为一个batch，据此计算梯度并更新参数。batch的大小一般取决于内存限制或者设置为2的幂。

#### Computing the Gradient

代数方法，求出损失函数对于每个参数的偏导数的代数式（反向传播）。

数值方法，一般用于调试和检验梯度的代数式是否正确，一般使用centered difference formula：
$$
[f(x+h) - f(x-h)]\ /\ 2h
$$

# TODO

#### Backpropagation (Backprop)

##### Derivatives, vectorization

### Feature

#### Color histogram

#### Histogram of Oriented Gradient (HOG)

### Convolutional neural networks (ConvNets)

#### Convolutional layer

#### Pooling layer
