# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:36:27 2022

@author: giorg
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 11:28:05 2022

@author: giorg
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 08:36:00 2022

@author: giorg
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:49:17 2022

@author: giorg
"""

import numpy as np
import pandas as pd
import os as os
import random
from statsmodels.discrete.discrete_model import Probit
from scipy.optimize import minimize
from numpy.random import rand
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import randint
from numpy.random import rand
from scipy.stats import norm
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm
from matplotlib import pyplot as plt
import statsmodels.api as sm
from linearmodels.panel import PooledOLS
from linearmodels.panel import BetweenOLS
from linearmodels.panel import PanelOLS

K=25358


reps=100
year1=1990
year2=2022


os.chdir("C:/Users/giorg/Desktop/Research/choice under risk experiment")
X= pd.read_csv("market.csv")

B=np.array(X[['vwretd','DATE']])
Date=np.array(X['DATE'])
year=Date//10000
month=(Date-(year*10000))//100
day=(Date-(year*10000)-(month*100))
A=np.zeros((K+1,7))       
A[:,0]=Date
A[:,1]=year
A[:,2]=month
A[:,3]=day
A[:,4]=B[:,0]

mean=np.zeros(1)
acorr=np.zeros(1)
var=np.zeros(1)
vol=np.zeros(1)
T=0

for i in range(year1,year2):
        z=A[(A[:,1]==i)]
        r=z[:,4]
        r_1=np.delete(r,-1,axis=0)
        r_minus0=np.delete(r,0,axis=0)
        rmean=np.mean(r)
        #rcorr=np.corrcoef(r_1,r_minus0)[0,1]
        rvar=np.var(r)
        rcorr=(np.cov(r_1,r_minus0)[0,1])/(rvar)
        acorr=np.append(acorr,[rcorr])
        var=np.append(var,[rvar])
        rvol=(np.sum((r-rmean)**2))**(1/2)
        vol=np.append(vol,[rvol])
        T=T+1
        
vol=vol[1:] 
acorr=acorr[1:]
var=var[1:]     
volqtile=np.quantile(vol,[0.25,0.5,0.75,0.9])
acorrqtile=np.quantile(acorr,[0.25,0.5,0.75,0.95])



pctileno=np.zeros(T)
pctile50=np.zeros(T)
pctile90=np.zeros(T)

pctile50_1=np.zeros(T)
pctile90_1=np.zeros(T)
#vol=np.transpose(np.array([vol,dvol,acorr]))


#vol = np.where(vol[:,1] > volqtile[1], 0, the_array)

Z=np.transpose(np.array([var,acorr,vol,pctile50,pctile90,pctileno,pctile50_1,pctile90_1]))
Z=pd.DataFrame(Z,columns=['var','autocorr','vol','pctile50','pctile90','pctileno','pctile50_1','pctile90_1'])
Z['pctile50']=np.where(Z['vol']>=volqtile[1],1,0)
Z['pctileno']=np.where(Z['vol']<volqtile[1],1,0)
Z['pctile90']=np.where(Z['vol']>=volqtile[3],1,0)
Z['pctile50']=np.where(Z['vol']>=volqtile[3],0,Z['pctile50'])

Z['pctile50_1']=np.where(Z['vol']>=volqtile[1],1,0)
Z['pctile90_1']=np.where(Z['vol']>=volqtile[3],1,0)

x=sm.add_constant(Z['var'])
mod=sm.OLS(Z['autocorr'],x)
result=mod.fit()
print(result.summary())
print(result.t_test([0,0]))

x2=sm.add_constant(Z['vol'])
mod2=sm.OLS(Z['autocorr'],x2)
result2=mod2.fit()
print(result2.summary())

x3=Z[['pctileno','pctile50','pctile90']]
mod3=sm.OLS(Z['autocorr'],x3)
result3=mod3.fit()
print(result3.summary())

x4=sm.add_constant(Z[['pctile50_1','pctile90_1']])
mod4=sm.OLS(Z['autocorr'],x4)
result4=mod4.fit()
print(result4.summary())
  

coeff_0=np.corrcoef(vol[:-1],acorr[1:])[0,1]

x0=sm.add_constant(vol[:-1])

mod0=sm.OLS(acorr[1:],x0)
result0=mod0.fit()
print(result0.summary())


p50=(1/2)*(((vol-volqtile[1])/np.abs(vol-volqtile[1]))+1)
p90=(1/2)*(((vol-volqtile[3])/np.abs(vol-volqtile[3]))+1)

s50=p50-p90 
s0=1-p50    
S_ptile=np.transpose(np.array([s0[:-1],s50[:-1],p90[:-1]]))
modstile=sm.OLS(acorr[1:],S_ptile)
resultstile=modstile.fit()
print(resultstile.summary())


X_ptile=np.transpose(np.array([p50[:-1],p90[:-1]]))
x_ptile=sm.add_constant(X_ptile)
modptile=sm.OLS(acorr[1:],x_ptile)
resultptile=modptile.fit()
print(resultptile.summary())





coeff=np.zeros(reps)
X=np.transpose(np.array([vol[:-1],acorr[1:]]))

for h in range(0,reps):
    W=np.zeros((T-1,2))
    for j in range(0,T-1):
       l=random.randint(0,T-2)
       W[j,:]=X[l,:]
    coeff[h]=np.corrcoef(W[:,0],W[:,1])[0,1]
   

sigma_c=(np.var(coeff))**(1/2)
CI_coeff=[coeff_0-1.96*(sigma_c),coeff_0+1.96*(sigma_c)]

vol_0=np.delete(vol,-1,axis=0)
    



