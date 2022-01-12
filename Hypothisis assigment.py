#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Hypothesis cutlets Question :


# In[158]:


import pandas as pd


# In[159]:


import scipy as sp


# In[160]:


import numpy as np


# In[161]:


import matplotlib.pyplot as plt


# In[162]:


import seaborn as sns


# In[163]:


from statsmodels.stats.proportion import proportions_ztest


# In[ ]:





# In[ ]:





# In[164]:


Cutlets= pd.read_csv("F:/Dataset/Cutlets.csv")


# In[165]:


Cutlets


# In[166]:


Cutlets.describe()


# In[167]:


unit_A= Cutlets["Unit A"].mean()


# In[168]:


unit_A


# In[169]:


unit_B= Cutlets["Unit B"].mean()


# In[170]:


unit_B


# In[175]:


print('Unit A > Unit B:',unit_A>unit_B)


# In[176]:


sns.boxplot(data=[Cutlets['Unit A'],Cutlets['Unit B']])
plt.legend(['Unit A','Unit B'])


# In[177]:


sns.distplot(Cutlets['Unit A'])
sns.distplot(Cutlets['Unit B'])
plt.legend(['Unit A','Unit B'])


# In[178]:


Unit_A = pd.DataFrame(Cutlets['Unit A'])
Unit_A


# In[179]:


Unit_B = pd.DataFrame(Cutlets['Unit B'])
Unit_B


# In[191]:


tStat,pValue =sp.stats.ttest_ind(Unit_A,Unit_B)


# In[193]:


sp.stats.ttest_ind(Unit_A,Unit_B)


# In[195]:


if pValue>0.05:
    print('accept null hypothesis')
else:
    print('dont accpetnull hypothesis ')


# In[110]:


'according to abouve analysis there is no significant diffrance in diameter of Unit A and Unit B'


# In[ ]:





# In[ ]:





# In[ ]:





# In[111]:


#Hypothesis Labtest question:


# In[196]:


import pandas as pd
import numpy as np


# In[113]:


Labtest= pd.read_csv("F:/Dataset/LabTAT.csv")


# In[114]:


Labtest


# In[115]:


Labtest.info()


# In[116]:


Labtest.describe()


# In[118]:


lab1=Labtest["Laboratory 1"].mean()


# In[119]:


lab2=Labtest["Laboratory 2"].mean()


# In[120]:


lab3=Labtest["Laboratory 3"].mean()


# In[121]:


lab4=Labtest["Laboratory 4"].mean()


# In[124]:


print("mean  for labtest=",lab1,lab2,lab3,lab4)


# In[127]:


print('lab1 > lab2 = ',lab1 > lab2)
print('lab2 > lab3 = ',lab2 > lab3)
print('lab3 > lab4 = ',lab3 > lab4)
print('lab4 > lab1 = ',lab4 > lab1)


# In[128]:


''' Its Null and alternatve testing 
There are no significant differences between the groups' mean Lab values. H0:μ1=μ2=μ3=μ4
There is a significant difference between the groups' mean Lab values. Ha:μ1≠μ2≠μ3≠μ4'''


# In[139]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[140]:


sns.distplot(Labtest["Laboratory 1"])
sns.distplot(Labtest["Laboratory 2"])
sns.distplot(Labtest["Laboratory 3"])
sns.distplot(Labtest["Laboratory 4"])
plt.legend(['Laboratory 1','Laboratory 2','Laboratory 3','Laboratory 4'])


# In[143]:


plt.figure(figsize=(15,10))
sns.boxplot(data=[Labtest['Laboratory 1'],Labtest['Laboratory 2'],Labtest['Laboratory 3'],Labtest['Laboratory 4']])
plt.legend(['Laboratory 1','Laboratory 2','Laboratory 3','Laboratory 4'])


# In[199]:


alpha=0.05
Lab1=pd.DataFrame(Labtest['Laboratory 1'])
Lab2=pd.DataFrame(Labtest['Laboratory 2'])
Lab3=pd.DataFrame(Labtest['Laboratory 3'])
Lab4=pd.DataFrame(Labtest['Laboratory 4'])


# In[145]:


print(Lab1,Lab2,Lab3,Lab4)


# In[149]:


tStat, pvalue = sp.stats.f_oneway(Lab1,Lab2,Lab3,Lab4)


# In[150]:


print("P-Value:{0} T-Statistic:{1}".format(pvalue,tStat))


# In[154]:


if pvalue<0.05:
        print("reject null hypothesis")
else:
    print("accept null Hypothesis")


# In[155]:


#according to abouve analysis there is no significant diffrance in avrage TaT for all labs


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


# Hypothisis Buyer ratio assigment 


# In[1]:


import pandas as pd
import numpy as np


# In[2]:


B_ratio= pd.read_csv("F:/Dataset/BuyerRatio.csv")


# In[3]:


B_ratio


# In[4]:


'''Assume Null Hypothesis as Ho: Independence of categorical variables male-female buyer rations are similar across regions (does not vary and are not related) 
Thus Alternate Hypothesis as Ha: Dependence of categorical variables male-female buyer rations are NOT similar across regions (does vary and somewhat/significantly related)'''


# In[5]:


abc=B_ratio.iloc[:,1:]


# In[6]:


abc


# In[8]:


import scipy as sp
from scipy import stats
from scipy.stats import norm
from scipy.stats import chi2_contingency


# In[11]:


chi2_contingency(abc)


# In[16]:


pValue = 0.66030


# In[17]:


''' Lets conmpaire Pvalue with 0.05'''


# In[18]:


if pValue<0.05:
    print('all preportion are equl')
else:
    print('all preportion are not equl')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


# Hypothesis Customerorder form assigment


# In[20]:


import pandas as pd 
import numpy as np


# In[21]:


Cust_order= pd.read_csv('F:/Dataset/Costomer+OrderForm.csv')


# In[22]:


Cust_order


# In[24]:


Cust_order.Phillippines.value_counts()


# In[25]:


Cust_order.Indonesia.value_counts()


# In[26]:


Cust_order.Malta.value_counts()


# In[27]:


Cust_order.India.value_counts()


# In[29]:


abc=np.array([[271,267,269,280],[29,33,31,20]])


# In[30]:


abc


# In[31]:


chi2_contingency(abc)


# In[32]:


pValue=0.28


# In[33]:


''' Lets conmpaire Pvalue with 0.05'''


# In[36]:


if pValue<0.05:
    print('we reject null hypothesis')
else:
    print('we accept null hypothesis')


# In[37]:


# above analysis shows that pValue is lesser than 0.05 thats why its showing we accept null hypothesis i.e customer order forms defective precentage does not varies by centre


# In[ ]:




