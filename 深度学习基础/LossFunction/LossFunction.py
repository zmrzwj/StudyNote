# -*-coding:utf-8-*-
import numpy as np

class LossFunction:

    def cosine_proximity(self, y_true, y_pre):
        #预测标签的与真实标签的余弦距离距离平均值的相反数

        y_true_square_sum_sqrt = np.sqrt(np.sum(np.square(y_true)))
        y_pre_square_sum_sqrt = np.sqrt(np.sum(np.square(y_pre)))
        y_dot = np.dot(y_true,y_pre)

        return y_dot/y_pre_square_sum_sqrt*y_true_square_sum_sqrt

    def expll(self, y_true, y_pre):
        #指数对数似然
        #Poisson回归，泊松回归
        return np.exp(-np.dot(y_true, y_pre))

    def hinge(self, y_true, y_pre):
        #铰链损失
        #主要用于SVM
        #预测值（-1到1之间），目标值（±1）
        return np.max(0, 1-y_true*y_pre)

    def kl_divergence(self, y_true, y_pre):
        #相对熵，KL散度
        #sum(px*log(px/qx))
        kl = 0
        for i in range(len(y_true)):
            kl = kl + y_true[i]*np.log(y_true[i]/y_pre[i])
        return kl

    def mcxent(self, y_true, y_pre):
        #多类交叉熵损失函数
        return y_true - y_pre

    def mean_absolute_error(self, y_true, y_pre):
        #平均绝对误差,MAE
        abs_ = []
        for i in range(len(y_true)):
            abs_[i] = np.abs(y_true[i]-y_pre[i])
        return np.sum(abs_)

    def mean_absolute_percentage_error(self, y_true, y_pre):
        # 平均绝对百分比误差,MAPE
        length = len(y_true)
        abs_ = []
        for i in range(len(y_true)):
            abs_[i] = np.abs((y_true[i]-y_pre[i])/y_true[i])*100
        return np.sum(abs_)/length

    def mse(self, y_true, y_pre):
        #均方误差损失函数
        pingfang = []
        for i in range(len(y_true)):
            pingfang[i] = np.square(y_true[i]-y_pre[i])
        return np.sum(pingfang)

    def msle(self, y_true, y_pre):
        #均方对数误差损失函数
        log_pingfang = []
        for i in range(len(y_true)):
            log_pingfang[i] = np.square(np.log(y_true[i]+1)-np.log(y_pre[i]))
        return np.sum(log_pingfang)/len(y_true)

    def negativeloglikelihood(self, y_true, y_pre):
        #负对数似然
        #将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min。
        #numpy.clip(a, a_min, a_max, out=None)
        #np.array可以直接加减
        #二分类损失
        eps = 1e-15
        y_t = np.array(y_true)
        y_p = np.array(y_pre)
        p = np.clip(y_p, eps, 1-eps)
        loss = np.sum(-y_t*np.log(p)-(1-y_t)*np.log(1-p))
        return loss / len(y_t)

    def poisson(self, y_true, y_pre):
        #即(predictions - targets * log(predictions))的均值
        #泊松损失
        return y_pre - y_true * np.log(y_true)

    def squared_hinge(self, y_true, y_pre):
        #简称L2 loss
        #(max(1-y_true*y_pred,0))^2.mean(axis=-1)，
        # 取1减去预测值与实际值乘积的结果与0比相对大的值的平方的累加均值
        max_ = []
        for i in range(len(y_true)):
            max_[i] = (np.max(0, 1-y_true*y_pre))**2
        return np.mean(max_)

    def logcosh(self, y_true, y_pre):
        #对数双曲余弦logcosh
        def cosh(x):
            return (np.exp(x)+np.exp(-x))/2
        log_ = []
        for i in range(len(y_true)):
            log_[i] = np.log(cosh(y_pre[i]-y_true[i]))
        return np.mean(log_)




if __name__ == "__main__":
    test = LossFunction()

    x = [1,2,3]

    x2 = [1,2,9]

    x_np = np.array(x)

    x2_np = np.array(x2)

    print x2_np - x_np

    print np.dot(x,x2)

