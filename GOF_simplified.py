import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#read the data into python
#df will be the data; df_dist_list is a helper to get the list of distributions
df = pd.read_csv('https://raw.githubusercontent.com/evangelistapaul/MC_EM384/main/IOCT_100.csv')
df_dist_list = pd.read_csv('https://raw.githubusercontent.com/evangelistapaul/MC_EM384/main/dist_list.csv', header = None)


#convert the data to a numpy array
x = np.array(df.iloc[:,0])

#visualize
plt.hist(x, bins=50)

#iterate through each distribution using a for loop and test the fit at each iteration:
results = []
for i in df_dist_list.iloc[:,0]:
    dist = getattr(stats, i)
    param = dist.fit(x)
    a = stats.kstest(x, i, args=param)
    results.append((i,a[0],a[1]))
    
#sort the list with largest p values at the top:
results.sort(key=lambda x:float(x[2]), reverse=True)

#let's use the lognormal distribution
dist = getattr(stats,'lognorm')
parameters = dist.fit(x)
print(parameters)
stats.kstest(x,"lognorm",parameters)



