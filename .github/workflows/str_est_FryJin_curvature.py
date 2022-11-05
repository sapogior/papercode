# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 07:17:13 2022

@author: giorg
"""

import numpy as np
import pandas as pd
import os as os
import scipy as sp
from statsmodels.discrete.discrete_model import Probit
from scipy.optimize import minimize
from scipy.stats import randint
from numpy.random import rand
from scipy.stats import norm
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


k=600
p=0.5
s=200
t=600
rho=0.8
theta=0
a=1
const=1
gamma=0.5
beta=0.5
param=np.array([rho,theta, gamma, beta])
num=200000
pbar=0.055
reps=250

os.chdir("https://github.com/sapogior/papercode/edit/main/.github/workflows")

exp= pd.read_stata("final200.dta")
Y=exp[['subj','trial','risky']]
#X=exp[['subj','trial','risky','x','c']]
A=exp[['subj','trial','x']]
X=exp[['subj','trial','x']]
C=exp[['subj','trial','c']]
#bnds = ((-0.000000001,0.00000001), (1.000000000, 1.00000001), (None,None)) 



def EU(param, X,C,p):
    gamma, beta = param
    xp=p*((np.array(X['x']))**(1-gamma))*beta
    c=((np.array(C['c']))**(1-gamma))*beta
    return xp-c
def LogLEU(param,Y,X,C,p):
    Y['U']=EU(param,X,C,p)
    Y=Y[~(Y['risky']==999)]
    y=np.array(Y['risky'])
    u=np.array(Y['U'])
    Like=(norm.cdf(u)**(y))*((1-norm.cdf(u))**(1-y))
    return -np.sum(np.log(Like))
H_EU_beta=minimize(LogLEU,[0,0], (Y,X,C,p),'Nelder-Mead')#,bounds=((None,None),(1,1.00001)))

def U(param, X,C,p,t,s):
    rho, theta, gamma, beta= param
    xp=p*((np.array(X['x']))**(1-gamma))
    c=((np.array(C['c']))**(1-gamma))
    pmean=np.zeros(t*s)
    for ind in range(1,s+1)  :  
        sozz=np.zeros(t+1) 
        azz=xp[(ind-1)*t:ind*t]
        for trial in range(1,t+1):  
            q=sozz.sum()
            mazz=azz[trial-1]
            sozz[trial-1]=mazz
            pmean[(ind-1)*t+trial-1]=q*(1-rho)+(rho**(trial-1))*sozz[trial-1] 
            sozz[0:trial-1]=rho*sozz[0:trial-1]
    D2=(xp-pmean)*(xp-pmean)*np.sign(xp-pmean)
    return beta*(pmean+theta*D2-c)

#rewrite U without dataframes
def V(param, X,C,p,t,s):
    rho, theta, gamma, beta= param
    xp=p*(X**(1-gamma))
    c=(C**(1-gamma))
    pmean=np.zeros(t*s)
    for ind in range(1,s+1)  :  
        sozz=np.zeros(t+1) 
        azz=xp[(ind-1)*t:ind*t]
        for trial in range(1,t+1):  
            q=sozz.sum()
            mazz=azz[trial-1]
            sozz[trial-1]=mazz
            pmean[(ind-1)*t+trial-1]=q*(1-rho)+(rho**(trial-1))*sozz[trial-1] 
            sozz[0:trial-1]=rho*sozz[0:trial-1]
    D2=(xp-pmean)*(xp-pmean)*np.sign(xp-pmean)
    return beta*(pmean+theta*D2-c)
#Z=U(param,X,C,p,t,s)
#Z=U(rho,theta,X,C,p,a,t,s)



def LogL(param,Y,X,C,p,t,s):
    Y['U']=U(param,X,C,p,t,s)
    Y=Y[~(Y['risky']==999)]
    y=np.array(Y['risky'])
    u=np.array(Y['U'])
    Like=(norm.cdf(u)**(y))*((1-norm.cdf(u))**(1-y))
    return -np.sum(np.log(Like))
#rewrite loglikelyhood in terms of V
def logl(param,Y,X,C,p,t,s):
    Y['U']=V(param,X,C,p,t,s)
    Y=Y[~(Y['risky']==999)]
    y=np.array(Y['risky'])
    u=np.array(Y['U'])
    Like=(norm.cdf(u)**(y))*((1-norm.cdf(u))**(1-y))
    return -np.sum(np.log(Like))

#LOL=LogL(param,Y,X,C,p,t,s)

Hbetozzo2=minimize(LogL,[0,0,0,0], (Y,X,C,p,t,s),'Nelder-Mead')#,bounds=((0,1),(None,None),(None,None),(1,1.01)))

W=np.zeros((t*s,3))
LOL=np.zeros((t*s,3))
LOL[:,0]=np.array(Y['risky'])
LOL[:,1]=np.array(X['x'])
LOL[:,2]=np.array(C['c'])
H1=np.zeros((reps,6))
j=np.zeros(s)
x80=np.array(X['x'])

#### block bootstrap ####

for o in range(1,reps+1):
    for i in range(1,s+1):
        l=randint.rvs(1,s)
        W[(i-1)*t:(i-1)*t+t,:]=LOL[int(l*t):int(l*t+t),:]
        j[i-1]=l

    w1=W[:,0]
    pboot=p
    w2=W[:,1]
    w3=W[:,2]
    Y1=pd.DataFrame(w1,columns=['risky'])    
   
    Q=minimize(logl,[0,0,0,0], (Y1,w2,w3,pboot,t,s), 'Nelder-Mead',bounds=((None,None),(None,None),(None,None),(None,None)))#,bounds=bnds2))
    H1[o-1,0:4]=Q['x']
    H1[o-1,4]=logl(Q['x'],Y1,w2,w3,pboot,t,s)
    H1[o-1,5]=Q['success']

rho, theta, gamma = Hbetozzo2['x'][0:3]
Hbuono=H1[H1[:,5]==1]
mirho=np.sum(Hbuono[:,0])/np.sum(Hbuono[:,5])
sigmarho=np.sum(Hbuono[:,0]*Hbuono[:,0])/np.sum(Hbuono[:,5])
mitheta=np.sum(Hbuono[:,1])/np.sum(Hbuono[:,5])-mirho**2
sigmatheta=np.sum(Hbuono[:,1]*Hbuono[:,1])/np.sum(Hbuono[:,5])-mitheta**2
migamma=np.sum(Hbuono[:,2])/np.sum(Hbuono[:,5])
sigmagamma=np.sum(Hbuono[:,2]*Hbuono[:,2])/np.sum(Hbuono[:,5])-migamma**2



CIrho=[rho-1.96*(sigmarho**(1/2)),rho+1.96*(sigmarho**(1/2))]
CItheta=[theta-1.96*(sigmatheta**(1/2)),theta+1.96*(sigmatheta**(1/2))]

Sigma=np.zeros((3,3))
Sigma[0,0]=sigmarho
Sigma[1,1]=sigmatheta
Sigma[0,1]=np.sum(Hbuono[:,0]*Hbuono[:,1])/np.sum(Hbuono[:,5])-mirho*mitheta
Sigma[1,0]=Sigma[0,1]
Sigma[2,2]=sigmagamma
Sigma[1,2]=np.sum(Hbuono[:,2]*Hbuono[:,1])/np.sum(Hbuono[:,5])-migamma*mitheta
Sigma[2,1]=Sigma[1,2]
Sigma[2,0]=np.sum(Hbuono[:,2]*Hbuono[:,0])/np.sum(Hbuono[:,5])-migamma*mirho
Sigma[0,2]=Sigma[2,0]

mi=np.array([rho,theta])

A=np.zeros((num,3))
for z in range(0,num):
    v=np.random.normal(loc=rho,scale=3)
    w=np.random.normal(loc=theta,scale=3)
    x=[v,w]
    Z=np.inner(x-mi,np.inner(inv(Sigma[0:2,0:2]),x-mi))
    A[z,0:2]=x
    A[z,2]=Z

A=A[A[:,2]<=5.99]
fig, ax = plt.subplots()

ax.scatter(A[:,0], A[:,1], c='lightblue')
ax.legend([],"hoho")
ax.scatter(rho,theta, c='blue')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\theta$')


N=50

pis = np.linspace(0.01, 0.1, num = N)
kstar = np.array([theta, gamma])
klow = np.array([CItheta[0],gamma])
khigh = np.array([CItheta[1],gamma])

def DisEU(pis,k):
    theta, gamma = k
    return pbar*(80**(1-gamma))+theta*(pis-pbar)**(2)*np.sign(pis-pbar)*(80**((1-gamma)*2))    
def TruEU(p):
    return p*80**(1-gamma)

plt.plot(pis, DisEU(pis,kstar), color='red')
plt.plot(pis, TruEU(pis), color='green')
#plt.plot(pis, Ubounds[0,:], color='red', linestyle='dashed')
#plt.plot(pis, Ubounds[1,:], color='red', linestyle='dashed')
#plt.plot(pis, Ubounds[2,:], color='blue', linestyle='dashed')
plt.plot(pis, DisEU(pis,klow), color='red', linestyle='--', linewidth=1)
plt.plot(pis, DisEU(pis,khigh), color='red', linestyle='--', linewidth=1)
plt.ylim(0,4)

