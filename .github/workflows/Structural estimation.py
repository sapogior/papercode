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

os.chdir("https://github.com/sapogior/papercode/edit/main/.github/workflows.py")

expa= pd.read_stata("T1_T2_T3_T4reshape.dta")
Ya=expa[['subj','trial','risky']]
#X=exp[['subj','trial','risky','x','c']]
Xa=expa[['subj','trial','x','P']]
Ca=expa[['subj','trial','c']]
expb= pd.read_stata("pilot25reshape.dta")
Yb=expb[['subj','trial','risky']]
#X=exp[['subj','trial','risky','x','c']]
Xb=expb[['subj','trial','x','P']]
Cb=expb[['subj','trial','c']]
pa=np.array(Xa['P'])
pb=np.array(Xb['P'])

bnds1 = ((None,None), (None,None))#(1,1.00000001))
bnds2 = ((None,None), (None, None), (0,0), (1,1))#(1,1.00000001))
param_EU_FJ=np.array([0.366261,1.38913])
param_FJ=np.array([0.40163, 1.33891, 0.18284, 0.835643])

#k=5
pbar=0.055
reps=10
ta=5
sa=1194
tb=4
sb=398
#rho=0.8
#theta=0
#a=1
num=50000

rhoFJ=0.4016
thetaFJ=1.3389

q1=0.6
param=np.array([0,0.5,0,1])

def EU(param, X,C,p):
    gamma, beta= param
    xp=p*((np.array(X['x']))**(1-gamma))*beta
    c=q1*((np.array(C['c']))**(1-gamma))*beta
    return xp-c
def LogLEU(param,Y,X,C,p):
    Y['U']=EU(param,X,C,p)
    Y=Y[~(Y['risky']==999)]
    y=np.array(Y['risky'])
    u=np.array(Y['U'])
    Like=(norm.cdf(u)**(y))*((1-norm.cdf(u))**(1-y))
    return -np.sum(np.log(Like))
#H_Sapo_EU_nobeta=minimize(LogLEU,[0], (Y,X,C,p),'Nelder-Mead')#, bounds=bnds1)


def U(param, X,C,p,t,s):
    rho, theta, gamma, beta = param
    xp=p*((np.array(X['x']))**(1-gamma))
    c=q1*((np.array(C['c']))**(1-gamma))
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

def V(param, X,C,p,t,s):
    rho, theta, gamma, beta = param
    xp=p*(X**(1-gamma))
    c=q1*(C**(1-gamma))
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



def LogL(param,Ya,Xa,Ca,pa,ta,sa,Yb,Xb,Cb,pb,tb,sb):
    Ya['U']=U(param,Xa,Ca,pa,ta,sa)
    Ya=Ya[~(Ya['risky']==999)]
    ya=np.array(Ya['risky'])
    ua=np.array(Ya['U'])
    Likea=(norm.cdf(ua)**(ya))*((1-norm.cdf(ua))**(1-ya))
    Yb['U']=U(param,Xb,Cb,pb,tb,sb)
    Yb=Yb[~(Yb['risky']==999)]
    yb=np.array(Yb['risky'])
    ub=np.array(Yb['U'])
    Likeb=(norm.cdf(ub)**(yb))*((1-norm.cdf(ub))**(1-yb))   
    return -(np.sum(np.log(Likea))+np.sum(np.log(Likeb)))
#rewrite loglikelyhood in terms of V

def logl(param,Ya,Xa,Ca,pa,ta,sa,Yb,Xb,Cb,pb,tb,sb):
    Ya['U']=V(param,Xa,Ca,pa,ta,sa)
    Ya=Ya[~(Ya['risky']==999)]
    ya=np.array(Ya['risky'])
    ua=np.array(Ya['U'])
    Likea=(norm.cdf(ua)**(ya))*((1-norm.cdf(ua))**(1-ya))
    Yb['U']=V(param,Xb,Cb,pb,tb,sb)
    Yb=Yb[~(Yb['risky']==999)]
    yb=np.array(Yb['risky'])
    ub=np.array(Yb['U'])
    Likeb=(norm.cdf(ub)**(yb))*((1-norm.cdf(ub))**(1-yb))   
    return -(np.sum(np.log(Likea))+np.sum(np.log(Likeb)))


#LOL=LogL(param,Y,X,C,p,t,s)

#for i in [0,1,2]:
#    for j in range 
#bloh=np.zeros((20,6))
#doh=np.linspace(0, 1, num=4)
#bah=np.linspace(0, 10, num=5)
#for i in range(0,4):
#    for j in range(0,5):
#        T1T4_nobeta=minimize(LogL,[doh[i],bah[j],0.5,1.0001], (Ya,Xa,Ca,pa,ta,sa,Yb,Xb,Cb,pb,tb,sb),'Nelder-Mead', bounds=((0,1),(None,None),(None,None),(1,1.001)))#,bounds=bnds2)
#        bloh[5*i+j,0:4]=T1T4_nobeta['x']
#        bloh[5*i+j,4]=LogL(T1T4_nobeta['x'],Ya,Xa,Ca,pa,ta,sa,Yb,Xb,Cb,pb,tb,sb)
#        bloh[5*i+j,5]=T1T4_nobeta['success']
 
T1T4_nobeta=minimize(LogL, [0,0,0,0], (Ya,Xa,Ca,pa,ta,sa,Yb,Xb,Cb,pb,tb,sb),'Nelder-Mead', bounds=((0,1),(None,None),(None,None),(None,None)))#,bounds=bnds2)
  
like_eu=LogLEU(param_EU_FJ,Yb,Xb,Cb,pb)+LogLEU(param_EU_FJ,Ya,Xa,Ca,pa)
like_fj=LogL(np.array([0.40163, 1.33891, 0.35284, 0.835643]),Ya,Xa,Ca,pa,ta,sa,Yb,Xb,Cb,pb,tb,sb)
    
Wa=np.zeros((ta*sa,3))
LOLa=np.zeros((ta*sa,3))
LOLa[:,0]=np.array(Ya['risky'])
LOLa[:,1]=np.array(Xa['P'])
LOLa[:,2]=np.array(Ca['c'])
x80a=np.array(Xa['x'])

Wb=np.zeros((tb*sb,3))
LOLb=np.zeros((tb*sb,3))
LOLb[:,0]=np.array(Yb['risky'])
LOLb[:,1]=np.array(Xb['P'])
LOLb[:,2]=np.array(Cb['c'])
x80b=np.array(Xb['x'])

H1=np.zeros((reps,6))
j=np.zeros(sa)
e=np.zeros(sb)

for o in range(1,reps+1):
    for i in range(1,sa+1):
        l=randint.rvs(1,(sa-1)+1)
        Wa[(i-1)*ta:(i-1)*ta+ta,:]=LOLa[int(l*ta):int(l*ta+ta),:]
        j[i-1]=l
    for d in range(1,sb+1):
        l=randint.rvs(1,(sb-1)+1)
        Wb[(d-1)*tb:(d-1)*tb+tb,:]=LOLb[int(l*tb):int(l*tb+tb),:]
        e[d-1]=l
    w1a=Wa[:,0]
    pboota=Wa[:,1]
    w3a=Wa[:,2]
    Y1a=pd.DataFrame(w1a,columns=['risky'])    
    w1b=Wb[:,0]
    pbootb=Wb[:,1]
    w3b=Wb[:,2]
    Y1b=pd.DataFrame(w1b,columns=['risky'])   
    Q=minimize(logl,[0,0,0,0], (Y1a,x80a,w3a,pboota,ta,sa,Y1b,x80b,w3b,pbootb,tb,sb), 'Nelder-Mead',bounds=((0,1),(None,None),(None,None),(None,None)))#,bounds=bnds2))
    H1[o-1,0:4]=Q['x']
    H1[o-1,4]=logl(Q['x'],Y1a,x80a,w3a,pboota,ta,sa,Y1b,x80b,w3b,pbootb,tb,sb)
    H1[o-1,5]=Q['success']

rho, theta, gamma = T1T4_nobeta['x'][0:3]
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
    v=np.random.normal(loc=mirho,scale=0.5)
    w=np.random.normal(loc=mitheta,scale=2)
    x=[v,w]
    Z=np.inner(x-mi,np.inner(inv(Sigma[0:2,0:2]),x-mi))
    A[z,0:2]=x
    A[z,2]=Z

A=A[A[:,2]<=5.99]
fig, ax = plt.subplots()

ax.scatter(A[:,0], A[:,1], c='lightblue')
ax.legend([],"hoho")
ax.scatter(rhoFJ,thetaFJ, c='red', label="mapensa")
ax.scatter(rho,theta, c='blue')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\theta$')


N=50
T=100


pis = np.linspace(-0.02, 0.13, num = N)
kstar = np.array([theta, gamma])
klow = np.array([CItheta[0],gamma])
khigh = np.array([CItheta[1],gamma])

def DisEU(pis,k):
    theta, gamma = k
    return pbar*(80**(1-gamma))+theta*(pis-pbar)**(2)*np.sign(pis-pbar)*(80**((1-gamma)*2))    
def TruEU(p):
    return p*80**(1-gamma)

Usim=np.zeros((T,N))
Ubounds=np.zeros((3,N))
    

#for j in range(0,T):
#    khat= np.random.multivariate_normal(kstar,Sigma[1:3,1:3])
#    Usim[j,:] = DisEU(pis,khat)
    #plt.plot(pis, DisEU(pis,[khat[0],gamma]), color='yellow', linestyle='dashed')

Usim2=np.append(Usim,np.zeros((T,1)),axis=1)

for i in range(0,N):
     Ubounds[0,i]= np.quantile(Usim[:,i], 0.975)
     Ubounds[1,i]= np.quantile(Usim[:,i], 0.025)
     Ubounds[2,i]= np.quantile(Usim[:,i], 0.5)
    
    
    

plt.plot(pis, DisEU(pis,kstar), color='red')
plt.plot(pis, TruEU(pis), color='green')
#plt.plot(pis, Ubounds[0,:], color='red', linestyle='dashed')
#plt.plot(pis, Ubounds[1,:], color='red', linestyle='dashed')
#plt.plot(pis, Ubounds[2,:], color='blue', linestyle='dashed')
plt.plot(pis, DisEU(pis,klow), color='red', linestyle='dashed')
plt.plot(pis, DisEU(pis,khigh), color='red', linestyle='dashed')
    
