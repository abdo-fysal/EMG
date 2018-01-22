
# coding: utf-8

# In[7]:

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from random import randint
from math import sqrt
class EMG:
    def __init__(self,data_path):
        
        self.path=data_path
        

    def get_data(self):
        data = open(self.path, 'r')

        data = data.readlines()
        x = 0
        t = []
        y = []

        for i in data:
           t.append(x)
           x = x + 1
           m = float(i.rstrip())
           y.append(m)
        return (y,t)
 
    def rectify(self,data):
        u=[abs(number) for number in data]
    
        return u

    

    

    
    def threshold(self,p):
        noise=np.array(p[0:100])
        
        return 3*np.std(noise)
            
    def get_muap_index(self,data_rectified,data,thr):
        l=len(data)
        T=20
        m=0
        n=m+T-1
        
        muap_values=[]
        muap_indexes=[]
        value=[]
        muap=[]

        while n<=l:
            b=[]
            c=[]
            x=data_rectified[m:n]
            t=sum(x)/len(x)
            xx=np.array(data[m:n])
            v=np.argmax(xx)+m
            v1=int(v-T/2)
            v2=int(v+T/2-1)

            if t>=thr :
                
                for i in range(20):
                    muap.append(v1+i)
                    c.append(v1+i)
                    b.append(data[v1+i])
                    value.append(1000)
                muap_values.append(b)
                muap_indexes.append(c)
                m=m+T
                n=n+T
   
            m=m+T
            n=n+T
        return (muap_indexes,muap_values,muap,value)
    def decomposition(self,muap_indexes,muap_values):
        muap_templates_indexes=[]
        muap_templates_values=[]

        muap_templates_indexes.append(muap_indexes[0])
        muap_templates_values.append(muap_values[0])
     
        g=[]
        flag=0
        for i in range(len(muap_values)):
            for j in range(len(muap_templates_values)):
            
                d=difference(muap_templates_values[j],muap_values[i])
                if d<=12.65**5:
                    g.append([i,j])
                    flag=1
            
            if flag==0:
                muap_templates_values.append(muap_values[i])
                muap_templates_indexes.append(muap_indexes[i])
                g.append([i,len(muap_templates_values)-1])
            else:
                flag=0
                    
        return (muap_templates_indexes,muap_templates_values,g)
    
    def difference(self,a,b):
        c=[x1 - x2 for (x1, x2) in zip(a, b)]
        c=[i**2 for i in c]
        return sum(c)
        
    def decomposition(self,muap_indexes,muap_values):
        muap_templates_indexes=[]
        muap_templates_values=[]

        muap_templates_indexes.append(muap_indexes[0])
        muap_templates_values.append(muap_values[0])
        g=[]
        flag=0
        for i in range(len(muap_values)):
            for j in range(len(muap_templates_values)):
            
                d=self.difference(muap_templates_values[j],muap_values[i])
                if d<=12.65**5:
                    g.append([i,j])
                    flag=1
            
            if flag==0:
                muap_templates_values.append(muap_values[i])
                muap_templates_indexes.append(muap_indexes[i])
            else:
                flag=0
                    
        return (muap_templates_indexes,muap_templates_values,g)
    def kmeans(self,muap_indexes,muap_values):
        u1=5
        u2=90
        u3=200
        u4=350
        d1=0
        d2=0
        d3=0
        d4=0
        k1_index=[]
        k2_index=[]
        k3_index=[]
        k4_index=[]
        k1_cluster=[]
        k2_cluster=[]
        k3_cluster=[]
        k4_cluster=[]
        k1_value=[]
        k2_value=[]
        k3_value=[]
        k4_value=[]
        k1_value_prev=[]
        k2_value_prev=[]
        k3_value_prev=[]
        k4_value_prev=[]
        
        
        for i in range(len(muap_indexes)):
            if i!=u1 and i!=u2 and i!=u3 and i!=u4:
                
            
                d1=self.difference(muap_values[u1],muap_values[i])
            
            if i!=u1 and i!=u2 and i!=u3 and i!=u4:
                d2=self.difference(muap_values[u2],muap_values[i])
                
            if i!=u1 and i!=u2 and i!=u3 and i!=u4:
                
                d3=self.difference(muap_values[u3],muap_values[i])
                
            if i!=u1 and i!=u2 and i!=u3 and i!=u4:   
            
                d4=self.difference(muap_values[u4],muap_values[i])
        
            d=min(d1,d2,d3,d4)
            
            if d==d1:
                k1_index.append(muap_indexes[i])
                k1_cluster.append(muap_values[i])
            elif d==d2:
                k2_index.append(muap_indexes[i])
                k2_cluster.append(muap_values[i])
            elif d==d3:
                k3_index.append(muap_indexes[i])
                k3_cluster.append(muap_values[i])
            elif d==d4:
                k4_index.append(muap_indexes[i])
                k4_cluster.append(muap_values[i])
        
        
        for i in range(20):
            s=0
            for j in range(len(k1_cluster)):
                s=s+k1_cluster[j][i]
            s=s/len(k1_cluster)
            k1_value.append(s)
        
        for i in range(20):
            s=0
            for j in range(len(k2_cluster)):
                s=s+k2_cluster[j][i]
            s=s/len(k2_cluster)
            k2_value.append(s)
                
                
        for i in range(20):
            s=0
            for j in range(len(k3_cluster)):
                s=s+k3_cluster[j][i]
            s=s/len(k3_cluster)
            k3_value.append(s)   
        
        for i in range(20):
            s=0
            for j in range(len(k4_cluster)):
                s=s+k4_cluster[j][i]
            s=s/len(k4_cluster)
            k4_value.append(s)   
            
        k1_value_prev=k1_value
        k2_value_prev=k2_value
        k3_value_prev=k3_value
        k4_value_prev=k4_value
             
   
        while(1) :
        
            for i in range(len(muap_indexes)):
            
                d1=self.difference(k1_value,muap_values[i])
            
                d2=self.difference(k2_value,muap_values[i])
                
                d3=self.difference(k3_value,muap_values[i])
            
                d4=self.difference(k4_value,muap_values[i])
        
                d=min(d1,d2,d3,d4)
            
                if d==d1:
                    k1_index.append(muap_indexes[i])
                    k1_cluster.append(muap_values[i])
                elif d==d2:
                    k2_index.append(muap_indexes[i])
                    k2_cluster.append(muap_values[i])
                elif d==d3:
                    k3_index.append(muap_indexes[i])
                    k3_cluster.append(muap_values[i])
                elif d==d4:
                    k4_index.append(muap_indexes[i])
                    k4_cluster.append(muap_values[i])
                
            k1_value=[]
            k2_value=[]
            k3_value=[]
            k4_value=[]
        
            for i in range(20):
                s=0
                for j in range(len(k1_cluster)):
                    s=s+k1_cluster[j][i]
                s=s/len(k1_cluster)
                k1_value.append(s)
        
            for i in range(20):
                s=0
                for j in range(len(k2_cluster)):
                    s=s+k2_cluster[j][i]
                s=s/len(k2_cluster)
                k2_value.append(s)
                
                
            for i in range(20):
                s=0
                for j in range(len(k3_cluster)):
                    s=s+k3_cluster[j][i]
                s=s/len(k3_cluster)
                k3_value.append(s)   
        
            for i in range(20):
                s=0
                for j in range(len(k4_cluster)):
                    s=s+k4_cluster[j][i]
                s=s/len(k4_cluster)
                k4_value.append(s)   
        
            if sqrt(self.difference(k2_value,k2_value_prev))<=0.01 and sqrt(self.difference(k1_value,k1_value_prev))<=0.01 and sqrt(self.difference(k3_value,k3_value_prev))<=0.01 and sqrt(self.difference(k4_value,k4_value_prev))<=0.01: 
                break
    
            k1_index=[]
            k2_index=[]
            k3_index=[]
            k4_index=[]
            k1_cluster=[]
            k2_cluster=[]
            k3_cluster=[]
            k4_cluster=[]
            k1_value_prev=k1_value
            k2_value_prev=k2_value
            k3_value_prev=k3_value
            k4_value_prev=k4_value
        return (k1_index,k1_cluster,k2_index,k2_cluster,k3_index,k3_cluster,k4_index,k4_cluster)
    
    
    
    
    


# In[10]:

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from math import sqrt
path = "Data.txt"
E=EMG(path)
y,t=E.get_data()
Y=E.rectify(y)
H=E.threshold(Y)
h=[H]*len(t)
m_index,m_value,m,n=E.get_muap_index(Y,y,H)
#m_t_in,m_t_v,g=decomposition(m_index,m_value)
colors=['red','blue','yellow','green','black','brown']
plt.xlim(30000, 35000)
plt.plot(t,y)
z=[[1000]*len(m_index[0])]*len(m_index)
k1_i,k1_v,k2_i,k2_v,k3_i,k3_v,k4_i,k4_v=E.kmeans(m_index,m_value)

yy=[]
w=[]
#w.append(g[0][0])



for i in range(20):
    yy.append(i)

#for i in range(len(g)):
    
  #  m=g[50][0]
  #  n=g[50][1]
    #if n not in w:
       # w.append(n)
        
    #plt.plot(yy,m_value[m],linestyle='-',color=colors[n])
        
    #plt.plot(m_index[m],z[m],"*",color=colors[n])

#plt.plot(yy,k4_v[0],linestyle='-',color=colors[3])
for i in range(len(k1_i)):
    plt.plot(k1_i[i],z[i],'*',color=colors[0])
for i in range(len(k2_i)):
    plt.plot(k2_i[i],z[i],'*',color=colors[1])
for i in range(len(k3_i)):
    plt.plot(k3_i[i],z[i],'*',color=colors[2])
for i in range(len(k4_i)):
    plt.plot(k4_i[i],z[i],'*',color=colors[3])
    
    

plt.show()





# In[ ]:



