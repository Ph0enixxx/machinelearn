import time
import random
import string
import numpy as np

sigmod = lambda x:1/(1+np.exp(-x)) 
#def sigmod(x):
#	return 1/(1+np.exp(-x))
def d(a,x):
	return x*(1-x)
#d = lambda fun,x,_min=0.0001:(fun(x+_min)-fun(x)/_min)
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])  
y = np.array([[0,1,1,1]]).T  
syn0 = 2*np.random.random((3,4)) - 1  
syn1 = 2*np.random.random((4,1)) - 1  

for i in range(100000):
	l1 = sigmod(np.dot(X,syn0))
	l2 = sigmod(np.dot(l1,syn1))
	l2_d = (y - l2)*d(sigmod,l2)
	l1_d = np.dot(l2_d,syn1.T) * d(sigmod,l1)
	syn1 += l1.T.dot(l2_d)
	syn0 += X.T.dot(l1_d)

print(l2)

#1.准备输入
#2.期望输出
#设定随机权值矩阵
#训练：调整权值
#将
# X = np.array([[0,0,2],[0,1,1],[1,0,1],[1,1,1]])#输入
# y = np.array([[0,1,1,0]]).T #期望输出

# syn0 = 2*np.random.random((3,4)) - 1 #第一层权重
# syn1 = 2*np.random.random((4,1)) - 1 #第二层权值

# #print(syn0)
# print(np.dot(X,syn0))

# for i in range(120000):
# 	l1 = 1/(1+np.exp(-(np.dot(X,syn0))))  #dot意为点乘 中间层l1为权重与输入乘积的sigmod函数值
# 	l2 = np.dot(l1,syn1)#1/(1+np.exp(-(np.dot(l1,syn1)))) #输出层与中间层的权值乘积

# 	l2_d = (y-l2) * 1#l2#(l2*(1-l2)) #误差函数：（期望-真实值）*   导数？？？
# 	if i%10000 == 0:
# 		print("l1:")
# 		print(l1)
# 		print("l2_d:")
# 		print(1-l2)
# 		print("l2*(1-l2):")
# 		print(l2*(1-l2))
# 	l1_d = l2_d.dot(syn1.T) * (l1 * (1-l1)) # 理解这里的矩阵乘法，简单的对应乘积 （fx*（1-fx））是sigmod的导数！！！
# 	syn1 += l1.T.dot(l2_d)
# 	syn0 += X.T.dot(l1_d)


# print(l2)
#更新权重：
"""
[[ 0.69903433],[ 0.72475143],[ 0.73705206],[ 0.7369908 ]]
梯度
http://money.finance.sina.com.cn/corp/go.php/vMS_MarketHistory/stockid/600030.phtml?year=2016&jidu=2

1.生成随机权重
2.输入信息
3.经过权重计算，输出
4.计算误差0.5*（误差）**2做和
5.求导的链式法则
#神经网络
1.链式求导法则:
设f、g为两个关于x可导的函数，则复合函数(f·g)(x)的导数为f'(g(x))g'(x)

---
deep learn and cnn 类似于训练的黑盒子，只关心输入输出以及精确度就行了。

强大之处在于利用网络中间某一层的输出当做是数据的另一层表达，作为学习的特征。

有效的关键是大规模的数据少量数据将无法将参数训练充分


卷积神经网络简介：

基本结构包括两层：1为特征提取层，每一个输入都与前一层的局部接受域相连，并提取该局部的特征。

其二是特征映射层，网络的每个计算层由多个特征映射组成   每个特征是一个平面，平面上所有神经元权值相等。


求导：
f(x,y) = xy

df/dx = y   df/dy = x


df(x)/dx = lim h->0 (f(x+h)-f(x))/h


"""



