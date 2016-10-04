#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tushare as ts

class Data(object):
	__slots__ = ('data')
	def __init__(self):
		self.data = pd.read_csv('601168.csv')
	def go(self):
		dataIp = self.data.loc[299:1:-1,['open','high','close','low','volume']]
		#获取『明天』的收盘价
		dataOp = self.data.loc[300:2:-1,'open']
		trainData = np.c_[dataIp.values,dataOp.values]
		return dataIp,dataOp#trainData
	def pre(self):
		
		pass
if __name__ == '__main__':
	a  =  Data()
	print type(a.go())