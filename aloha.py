#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import tushare as ts
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from Data import Data

_d = Data()
In,Out = _d.go()
In = sm.add_constant(In)
print In.values
print Out.values.T
results = sm.OLS(Out.values.T,In.values).fit()
print(results.summary())
print(help(results))
#回归完成 下面来验证
plt.plot(Out.values,'b-',label='normal')
plt.plot(results.fittedvalues,'r--',label="ols")
plt.show()

