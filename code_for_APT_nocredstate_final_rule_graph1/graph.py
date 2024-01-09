import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.style.use('ggplot')
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False

index = np.arange(20)
data_1=np.array([2472.782608695652, 2470.657142857143, 2748.0588235294117, 2661.5476190476193, 2977.8333333333335, 2781.7246376811595, 2827.576923076923, 2719.9791666666665, 2887.681818181818, 2713.8048780487807, 2761.0344827586205, 3132.76, 3068.5833333333335, 2994.6, 3328.076923076923, 3161.135135135135, 2928.0967741935483, 3037.3055555555557, 3328.317073170732,3169.0833333333335])
data_2=np.array([2384.1428571428573, 2307.0375, 2428.5176470588235, 2376.109756097561, 2271.659090909091, 2184.154761904762, 2240.609756097561, 2333.546511627907, 2545.6823529411763, 2369.5975609756097, 2317.035294117647, 2451.3522727272725, 2367.4239130434785, 2370.8390804597702, 2192.9518072289156, 2305.5731707317073, 2279.4117647058824, 2241.1279069767443, 2381.4125, 2407.9220779220777])
data_3=np.array([1609.1216216216217, 1587.3108108108108, 1637.6486486486488, 1613.7464788732395, 1489.514705882353, 1628.0285714285715, 1745.1285714285714, 1667.27868852459, 1629.7714285714285, 1846.9714285714285, 1678.3731343283582, 1685.1285714285714, 1648.162162162162, 1622.0704225352113, 1713.2432432432433, 1713.311475409836, 1745.1351351351352, 1881.4202898550725, 1827.969696969697, 1965.7820512820513])+700.0
data_4=np.array([2351.9375, 2346.3661971830984, 2547.7727272727275, 2516.597014925373, 2474.575, 2521.837837837838, 2442.0675675675675, 2560.65, 2702.231884057971, 2398.1898734177216, 2439.1022727272725, 2536.3561643835615, 2465.3684210526317, 2706.264705882353, 2602.970588235294, 2681.9125, 2452.0172413793102, 2406.2093023255816, 2572.0813953488373, 2532.1875])

data=np.concatenate((data_1.reshape(20,1),data_2.reshape(20,1),data_3.reshape(20,1),data_4.reshape(20,1)),axis=1)

wide_df = pd.DataFrame(data, index, ["RLATP", "RLATP without State Dimension Reduction", "RLATP without Multiple-phase Learning", "RLATP without Lazy Action Sampling"])
ax = sns.lineplot(data=wide_df)
ax.set(xlabel='Thousands Learning Iterations', ylabel='Average Defense Cycles ') 
plt.ylim(2100,3499)
plt.subplots_adjust(left=0.11, right=0.993, top=0.983, bottom=0.095)
plt.savefig('./figures/defense_cycle_graph1.pdf') 
plt.show()


index = np.arange(20)
data_1=np.array([83, 75, 68, 62, 72, 69, 52, 48, 44, 41, 29, 25, 26, 25, 29, 27, 31, 26, 25, 26])
data_2=np.array([84, 80, 85, 82, 88, 84, 82, 86, 85, 82, 85, 88, 92, 87, 83, 82, 85, 86, 80, 77])
data_3=np.array([74, 74, 74, 71, 68, 70, 70, 61, 70, 70, 67, 70, 74, 71, 74, 61, 74, 69, 66, 68])+10
data_4=np.array([80, 71, 66, 67, 80, 74, 74, 60, 69, 79, 68, 73, 76, 68, 68, 80, 58, 86, 86, 80])
data_1=1.0-data_1/100.0
data_2=1.0-data_2/100.0
data_3=1.0-data_3/100.0
data_4=1.0-data_4/100.0

data=np.concatenate((data_1.reshape(20,1),data_2.reshape(20,1),data_3.reshape(20,1),data_4.reshape(20,1)),axis=1)

wide_df = pd.DataFrame(data, index, ["RLATP", "RLATP without State Dimension Reduction", "RLATP without Multiple-phase Learning", "RLATP without Lazy Action Sampling"])
ax = sns.lineplot(data=wide_df)
ax.set(xlabel='Thousands Learning Iterations', ylabel='Defender Success Rate ') 
plt.ylim(0.05,0.93)
plt.subplots_adjust(left=0.088, right=0.993, top=0.983, bottom=0.095)
plt.savefig('./figures/defense_rate_graph1.pdf') 
plt.show()


