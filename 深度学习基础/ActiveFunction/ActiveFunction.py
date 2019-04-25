# -*- coding: utf-8 -*-
import numpy as np

class ActiveClass:

    def cube(self, x):
        # x^3
        return x * x * x

    def elu(self, x, a):
        #指数线性单元
        #a是一个可调整的参数，它控制着ELU负值部分在何时饱和,a要大于0.

        if x >= 0:
            return x
        else:
            return a*(np.exp(x) - 1)

    def relu(self, x):
        #线性整流函数
        if x < 0:
            return 0
        else:
            return x

    def leaky_relu(self, x, a):
        #带泄露线性整流函数
        #a>0
        # a初始化为0.2,

        if x > 0:
            return x
        else:
            return a*x

    def PRelu(self, x, a):
        #参数线性整流（Parametric ReLU）
        #a为一个可通过反向传播算法（Backpropagation）学习的变量。
        # 当a很小且固定，PRelu退化为leaky_relu
        # a初始化为0.25
        if x > 0:
            return x
        else:
            return a*x

    def RRelu(self, x, a):
        #带泄露随机线性整流（Randomized Leaky ReLU,RReLU）
        #a是一个取自连续性均匀分布的随机变量，U(l,u),l<u,l、u为[0,1)
        if x > 0:
            return x
        else:
            return a*x

    def relu6(self, x):
        #这个函数可以将激励函数的值数据位于0~6之间
        if x < 0:
            return 0
        if x > 6:
            return 6
        else:
            return x


    def sigmoid(self, x):
        #激活函数计算量大，反向传播求误差梯度时，求导涉及除法
        #反向传播时，很容易就会出现梯度消失的情况，从而无法完成深层网络的训练
        #Sigmoids函数饱和且kill掉梯度。
        #Sigmoids函数收敛缓慢。
        return 1/(np.exp(-x)+1)

    def hard_sigmoid(self, x):
        #实际上就是如果<=-1输出为0，>=1输出为1，中间为一个线性
        return np.max(0, np.min(1, (1+x)/2))


    def softmax(self, x):
        #x必须为向量
        #softmax计算方便
        #
        ex = np.exp(-x)
        return ex/np.sum(ex)

    def softmax_(self, x):
        #数值稳定的softmax版本
        x = x - np.max(x)
        ex = np.exp(x)
        return ex/np.sum(ex)

    def softplus(self, x):
        #softplus函数可以用来产生正态分布的β和σ参数
        #softplus 是对 ReLU 的平滑逼近的解析函数形式。
        #np.log以e为低数
        return np.log(1+np.exp(x))

    def softsign(self, x):
        #tanh比softsign更容易饱和
        #Softsign 是 Tanh 激活函数的另一个替代选择。就像 Tanh 一样，Softsign 是反对称、去中心、可微分，并返回-1 和 1 之间的值。
        # 其更平坦的曲线与更慢的下降导数表明它可以更高效地学习，比tanh更好的解决梯度消失的问题。另一方面，导数的计算比 Tanh 更麻烦
        #在实践中，可以深度用softsign替代tanh激活函数
        return x/(1+np.abs(x))

    def tanh(self, x):
        #双曲正切函数
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    def rectifiedtanh(self, x):
        #max(0, tanh(x))
        #整流双曲正切函数
        return np.max(0, self.tanh(x))

    def selu(self, r=1.0507009873554804934193349852946, x=0, a=1.6732632423543772848170429916717):
        #使用该激活函数后使得样本分布满足零均值和单位方差
        if x > 0:
            return r*x
        else:
            return r*a*(np.exp(x)-1)


    def swish(self, x):
        #自门控激活函数
        #Swish 无上界有下界
        #Swish 是平滑且非单调的函数
        #自门控的优势是它仅需要一个简单的标量输入，而正常的门控需要多个标量输入。
        # 该特性令使用自门控的激活函数如Swish能够轻松替换以单个标量作为输入的激活函数(如ReLU)无需改变参数的隐藏容量或数量。
        return x*self.sigmoid(x)

    def thresholdedrelu(self, x, theta=1.0):
        #带有门限的ReLU
        if x > theta:
            return x
        else:
            return 0

    def maxout(self, x, w, b):
        #x为输入向量，w为权重，b为偏置
        #w和b的产生可以随机也可以产生自某种分布
        #Maxout的拟合能力非常强，可以拟合任意的凸函数。
        #Maxout具有ReLU的所有优点，线性、不饱和性。
        #同时没有ReLU的一些缺点。如：神经元的死亡。
        #每个神经元中有两组(w,b)参数，那么参数量就增加了一倍，这就导致了整体参数的数量激增。
        return np.max(np.dot(x, w) + b)


if __name__ == "__main__":
    test = ActiveClass()

    print "cube:", test.cube(2.3)
    print "elu:", test.elu(2.4, 0.5)