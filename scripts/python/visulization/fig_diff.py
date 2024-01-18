#%%
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
#%%
# read data
data = pd.DataFrame(columns=['Mean Relative Absolute Error','Data Type','Method'])
d = np.loadtxt("../result/result_1.txt")
m,_ = d.shape
for i in range(m):
    data.loc[len(data)] = [np.log(d[i,0]+1e-5),'Linear','WET']
d = np.loadtxt("../result/result_1_comp.txt")
m = d.shape[0]
for i in range(m):
    data.loc[len(data)] = [np.log(1e-5+d[i]),'Linear','GEP']

d = np.loadtxt("../result/result_2.txt")
m,_ = d.shape
for i in range(m):
    data.loc[len(data)] = [ np.log(d[i,0]+1e-5),'Quadratic','WET']
d = np.loadtxt("../result/result_2_comp.txt")
m = d.shape[0]
for i in range(m):
    data.loc[len(data)] = [np.log(1e-5+d[i]),'Quadratic','GEP']

d = np.loadtxt("../result/result_3.txt")
m,_ = d.shape
for i in range(m):
    data.loc[len(data)] = [np.log(d[i,0]+1e-5),'MM','WET']
d = np.loadtxt("../result/result_3_comp.txt")
m = d.shape[0]
for i in range(m):
    data.loc[len(data)] = [np.log(1e-5+d[i]),'MM','GEP']

d = np.loadtxt("../result/result_4.txt")
m,_ = d.shape
for i in range(m):
    data.loc[len(data)] = [np.log(d[i,0]+1e-5),'Hill','WET']
d = np.loadtxt("../result/result_4_comp.txt")
m = d.shape[0]
for i in range(m):
    data.loc[len(data)] = [np.log(1e-5+d[i]),'Hill','GEP']
# %%
sns.set(context='notebook', style='ticks',font_scale=1.4)
fig = sns.violinplot(x = 'Data Type',y = 'Mean Relative Absolute Error',
               hue = 'Method',data=data,split=False,
               inner="box", scale="count",cut = 0)
# %%
fig_ = fig.get_figure()
fig_.savefig("fig_compare.png",dpi=400)
# %%
