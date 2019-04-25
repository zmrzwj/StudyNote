# -*- coding: utf-8 -*-
import numpy as np

class InitWeightFunction:
    def distribution(self, distribution, x):
        #distribution为给定分布，len(x)为具体大小
        #表明x产生自给定分布
        #numpy.random提供了特定的分布
        #numpy.random.normal(均值,方差);正太分布
        # numpy.random.uniform(上界,下届);均匀分布
        # numpy.random.possion(λ);泊松分布
        #dirichlet狄利克雷分布,binomial二项分布,chisquare卡方分布,exponential指数分布,fF分布样本
        #multinomial多项分布,laplace拉普拉斯或双指数分布样本等共35种分布。
        return distribution(len(x))


    def identity(self, x):
        #单位矩阵初始化
        return x

    def zero(self, x):
        #全为0
        return np.zeros(len(x))

    def lecun_normal(self, x):
        #LeCun正态分布初始化方法
        #0均值，标准差为stddev = sqrt(1 / fan_in)的正态分布产生
        return np.random.normal(0,np.sqrt(1/len(x)),len(x))

    def lecun_uniform(self, x):
        #LeCun均匀分布初始化方法
        # [-a,a]的均匀分布
        #a=sqrt(3 / fan_in), fin_in是权重向量的输入单元数（扇入）
        a=np.sqrt(3/len(x))
        return np.random.uniform(-a, a, len(x))


    def normal(self, x, mean, stdev):
        #mean均值，stdev标准差
        return np.random.normal(mean, stdev, len(x))


    def ones(self, x):
        #全为1
        return np.ones(len(x))


    def uniform(self, x):
        #a=1/sqrt(fanIn)
        a = 1/np.sqrt(len(x))
        return np.random.uniform(-a, a, len(x))

    def xavier(self, len_in, len_out):
        #Gaussian distribution mean 0, variance 2.0/(fanIn + fanOut)
        #len_in输入神经元个数，len_out输出神经元个数
        mean = 0
        var = 2/(np.sqrt(len_in+len_out))
        return np.random.normal(mean, var, len_in)

    def xavier_uniform(self, len_in, len_out):
        #Uniform distribution U(-s,s) with s = sqrt(6/(fanIn + fanOut))
        s = np.sqrt(6/(len_in+len_out))
        return np.random.uniform(-s, s, len_in)











if __name__ == "__main__":
    test = InitWeightFunction()