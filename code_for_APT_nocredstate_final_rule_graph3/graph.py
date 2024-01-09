import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.style.use('ggplot')
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False

index = np.arange(20)
data_1=np.array([1617.5757575757575, 2319.1944444444443, 2212.813953488372, 2355.025641025641, 2352.813953488372, 2233.3888888888887, 2523.025641025641, 2410.4827586206898, 2847.4166666666665, 2761.0588235294117, 2829.1481481481483, 2981.5882352941176, 2679.0, 2867.2916666666665, 3112.21875, 2927.655172413793, 3065.56, 3096.6, 3200.40625, 3106.6470588235293])
data_2=np.array([1683.6029411764705, 1563.8529411764705, 1616.7611940298507, 1353.8684210526317, 1470.264705882353, 1476.7307692307693, 1658.774647887324, 1693.8823529411766, 1533.3030303030303, 1692.8648648648648, 1719.8333333333333, 1714.5714285714287, 1505.835616438356, 1624.5, 1428.4788732394366, 1685.527027027027, 1648.623188405797, 1707.2835820895523, 1663.144927536232, 1556.7808219178082])
data_3=np.array([1609.1216216216217, 1587.3108108108108, 1637.6486486486488, 1713.7464788732395, 1489.514705882353, 1628.0285714285715, 1845.1285714285714, 1667.27868852459, 1629.7714285714285, 1846.9714285714285, 1678.3731343283582, 1685.1285714285714, 1648.162162162162, 1622.0704225352113, 1713.2432432432433, 1713.311475409836, 1745.1351351351352, 1881.4202898550725, 1827.969696969697, 1965.7820512820513])
data_4=np.array([1495.9130434782608, 1835.0, 2066.491525423729, 2090.135593220339, 1853.9473684210527, 2087.705882352941, 2131.843137254902, 2352.3387096774195, 2163.6363636363635, 2128.423728813559, 2111.021739130435, 2325.1296296296296, 2146.0158730158732, 1756.7592592592594, 2001.6451612903227, 1843.5357142857142, 1661.0566037735848, 2054.6774193548385, 2485.109090909091, 1995.4716981132076])

data=np.concatenate((data_1.reshape(20,1),data_2.reshape(20,1),data_3.reshape(20,1),data_4.reshape(20,1)),axis=1)

wide_df = pd.DataFrame(data, index, ["RLATP", "RLATP without State Dimension Reduction", "RLATP without Multiple-phase Learning", "RLATP without Lazy Action Sampling"])
ax = sns.lineplot(data=wide_df)
ax.set(xlabel='Thousands Learning Iterations', ylabel='Average Defense Cycles ') 
plt.ylim(1300,3499)
plt.subplots_adjust(left=0.11, right=0.993, top=0.983, bottom=0.095)
plt.savefig('./figures/defense_cycle_graph3.pdf') 
plt.show()


index = np.arange(20)
data_1=np.array([66, 52, 43, 39, 43, 36, 39, 49, 36, 34, 27, 34, 23, 24, 32, 29, 25, 23, 28, 24])
data_2=np.array([68, 68, 67, 76, 68, 78, 71, 68, 66, 74, 66, 70, 73, 70, 71, 74, 69, 67, 69, 73])
data_3=np.array([74, 74, 74, 71, 68, 70, 70, 61, 70, 70, 67, 70, 74, 71, 74, 61, 74, 69, 66, 78])
data_4=np.array([69, 60, 59, 59, 57, 51, 51, 62, 55, 59, 46, 54, 63, 54, 62, 56, 53, 62, 55, 53])
data_1=1.0-data_1/100.0
data_2=1.0-data_2/100.0
data_3=1.0-data_3/100.0
data_4=1.0-data_4/100.0

data=np.concatenate((data_1.reshape(20,1),data_2.reshape(20,1),data_3.reshape(20,1),data_4.reshape(20,1)),axis=1)

wide_df = pd.DataFrame(data, index, ["RLATP", "RLATP without State Dimension Reduction", "RLATP without Multiple-phase Learning", "RLATP without Lazy Action Sampling"])
ax = sns.lineplot(data=wide_df)
ax.set(xlabel='Thousands Learning Iterations', ylabel='Defender Success Rate ') 
plt.ylim(0.2,0.89)
plt.subplots_adjust(left=0.088, right=0.993, top=0.983, bottom=0.095)
plt.savefig('./figures/defense_rate_graph3.pdf') 
plt.show()


