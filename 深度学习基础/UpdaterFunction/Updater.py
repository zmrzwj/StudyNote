# -*-coding:utf-8 -*-
import numpy as np

class Updater:

    def sgd(self, w, dw, learning_rate = 1e-2):
        #随机梯度，mini-batch gradient descent
        #缺点：选择合适的学习速率较难；容易局部收敛；稀疏数据或特征无法加快更新
        w = w - learning_rate * dw
        return w

    def AdaGrad(self, w, dw):
        #阿达格雷
        #learning_rate学习速率，Gt对角矩阵，每个对角线位置i,i的值累加到t次迭代的对应参数 θi 梯度平方和，ϵ是平滑项防止除0
        learning_rate = 1e-2
        epsilon = 1e-8
        cache = np.zeros_like(w)
        cache = cache + dw**2
        w = w - learning_rate * dw /(np.sqrt(cache)+epsilon)
        return w

    def RMSprop(self, w, dw):
        #RMSProp是AdaGrad的升级版本
        #RMSProp增加了一个衰减系数来控制历史信息的获取多少
        #依然依赖全局学习速率
        #适合处理非平稳目标；对RNN效果好
        #decay_rate衰减速率，learning_rate学习速率，epsilon是平滑项防止除0
        learning_rate = 1e-2
        decay_rate = 0.99
        epsilon = 1e-8
        cache = np.zeros_like(w)
        cache = cache * decay_rate + (1-decay_rate)*dw**2
        w = w - learning_rate * dw / (np.sqrt(cache + epsilon))
        return w

    def adam(self, w, dw):
        #阿达姆
        #带动量的RMSprop
        #参数：beta1 一阶矩估计的指数衰减率，0.9；beta2：二阶矩估计的指数衰减率0.999;epsilon：该参数是非常小的数，其为了防止在实现中除以零;
        #learning_rate学习速率0.001
        learning_rate = 1e-3
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        m = np.zeros_like(w) #一阶矩变量
        v = np.zeros_like(w) #二阶矩变量
        t = 0 #时间步

        m = m*beta1 + (1-beta1)*dw #更新有偏一阶矩估计
        v = v*beta2 + (1-beta2)*dw**2 #更新有偏二阶矩估计
        t = t + 1

        m_ = m / (1 - beta1**t) #修正一阶矩的偏差
        v_ = v / (1 - beta2**t) #修正二阶矩的偏差

        w = w - learning_rate * m_ / (np.sqrt(v_) + epsilon)
        return w

    def AdaDelta(self, w, dw, Eg):
        #阿达德尔塔
        #自适应调整学习率的方法
        #AdaDelta是对AdaGrad的扩展
        #初中期，加速效果不错，很快；后期，反复在局部最小值附近抖动
        #Eg 上一时刻的梯度平均值
        learning_rate = 0.001
        p = 0.9 #动量因子
        epsilon = 1e-8
        Eg = p*Eg+(1-p)*dw**2
        w = w - learning_rate*dw/(np.sqrt(Eg)+epsilon)
        return w

    def adamax(self, w, dw):
        #Adam的一种变体，此方法对学习速率上限提供了一个简单的范围
        return w

    def nadam(self, w, dw):
        #带有nesterov动量项的Adam
        #Nadam对学习率有了更强的约束，同时对梯度的更新也有更直接的影响。
        # 一般而言，在想使用带动量的RMSprop，或者Adam的地方，大多可以使用Nadam取得更好的效果。
        return w

    def nesterovs(self, w, dw):
        #学习速率 ϵ,  初始速率v, 动量衰减参数α
        learning_rate = 1e-2 #学习速率
        v = 0 #初始速率
        alpha = 0.9 #动量参数

        v = alpha*v - learning_rate*dw
        w = w + v
        return w




if __name__ == "__main__":
    test = Updater()