# coding:utf-8
"""https://www.cnblogs.com/qscqesze/p/6530142.html

author:@qscqesze"""
from numpy import exp,array,random,dot


class NeuralNetwork():
    def __init__(self):
        # 生成随机数种子
        random.seed(1)
        # 对每个神经元建模，含有三个输入和一个输出连接
        # 对3 * 1的矩阵赋予随机权重值，范围[-1,1],平均数为0
        self.synaptic_weights = 2 * random.random((3,1)) - 1

    # sigmoid 函数
    # 正规化操作，使得每个元素都是0~1
    def __sigmoid(self,x):
        return 1 / (1 + exp(-x))

    # sigmoid 函数求导
    # sigmoid 函数梯度
    # 表示我们对当前权重的置信程度
    def __sigmoid_derivative(self,x):
        return x * (1-x)

    # 神经网络——思考
    def think(self,inputs):
        # 把输入传递给神经网络
        return self.__sigmoid(dot(inputs,self.synaptic_weights))

    # 神经网络
    def train(self,training_set_inputs,training_set_outputs,number_of_training):
        for iteration in xrange(number_of_training):
            # 训练集导入神经网络
            output = self.think(training_set_inputs)

            # 计算误差
            error  = training_set_outputs - output

            # 将误差、输入和S曲线相乘
            # 对于置信程度低的权重，调整程度也越大
            # 为0的输入值不会影响权重
            adjustment = dot(training_set_inputs.T,error * self.__sigmoid_derivative(output))

            # 调整权重
            self.synaptic_weights += adjustment

if __name__ == "__main__":

    # 初始化神经网络
    neuralnetwork = NeuralNetwork()

    print("训练前的权重")
    print(neuralnetwork.synaptic_weights)

    # 训练集，四个样本，3个输入，1个输出
    training_set_inputs = array([[0,0,1],
                                 [1,1,1],
                                 [1,0,1],
                                 [0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T

    # 训练神经网络
    # 10000次训练

    neuralnetwork.train(training_set_inputs,training_set_outputs,10000)

    print("训练后的权重")
    print(neuralnetwork.synaptic_weights)

    # 新数据测试

    print("考虑[1,0,0]")
    print(neuralnetwork.think(array([1,0,0])))
