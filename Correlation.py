#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import kendalltau,spearmanr
from scipy import stats


# In[33]:


# Read the text file
path=r"C:\Users\shafaattahir\Downloads\po1_data.txt"
with open(path, 'r') as file:
    data = file.readlines()


# In[34]:


names=['Subject identifier This number identifies a study subject',
'Jitter Jitter in %',
'Jitter Absolute jitter in microseconds',
'Jitter Jitter as relative amplitude perturbation (r.a.p.)',
'Jitter Jitter as 5-point period perturbation quotient (p.p.q.5)',
'Jitter Jitter as average absolute difference of differences between jitter cycles'
'(d.d.p.)',
'Shimmer Shimmer in %',
'Shimmer Absolute shimmer in decibels (dB)',
'Shimmer Shimmer as 3-point amplitude perturbation quotient (a.p.q.3)',
'Shimmer Shimmer as 5-point amplitude perturbation quotient (a.p.q.5)',
'Shimmer Shimmer as 11-point amplitude perturbation quotient (a.p.q.11)',
'Shimmer Shimmer as average absolute differences between consecutivedifferences between the amplitudes of shimmer cycles (d.d.a.)',
'Harmonicity Autocorrelation between NHR and HNR',
'Harmonicity Noise-to-Harmonic ratio (NHR)',
'Harmonicity Harmonic-to-Noise ratio (HNR)',
'Pitch Median pitch',
'Pitch Mean pitch',
'Pitch Standard deviation of pitch',
'Pitch Minimum pitch',
'Pitch Maximum pitch',
'Pulse Number of pulses',
'Pulse Number of periods',
'Pulse Mean period',
'Pulse Standard deviation of period',
'Voice Fraction of unvoiced frames',
'Voice Number of voice breaks',
'Voice Degree of voice breaks',
'UPDRS The Unified Parkinson’s Disease Rating Scale (UPDRS) score that isassigned to the subject by a physician via a medical examination todetermine the severity and progression of Parkinson’s disease.',
'PD indicator' ]


# In[35]:


len(names)


# In[36]:


import pandas as pd

# Read the text file into a dataframe
df = pd.read_csv(path, delimiter=',',names=names
                                           )

# Display the dataframe
print(df)


# In[37]:


(df.columns)


# In[38]:


df.head()  # Display the first few rows of the dataset


# In[39]:


df.info()  # Display information about the dataset, such as data types and missing values


# In[40]:


df.describe()  # Summary statistics for numerical columns


# In[41]:


df.isnull().sum()  # Check for missing values in each column
df = df.dropna()  # Drop rows with missing values


# In[42]:


# make 2 dataframe 
pd_no = df[df['PD indicator'] == 0]
# pd_no.to_csv(r'C:\Users\shafaattahir\Downloads\no.csv')
pd_yes = df[df['PD indicator'] == 1]
# pd_yes.to_csv(r'C:\Users\shafaattahir\Downloads\yes.csv')


# In[43]:


# drop PD indicator column
pd_no=pd_no.drop('PD indicator',axis=1)
pd_yes=pd_yes.drop('PD indicator',axis=1)

# we will drop ''UPDRS The Unified Parkinson’s Disease Rating Scale (UPDRS) score that isassigned to the subject by a physician via a medical examination todetermine the severity and progression of Parkinson’s disease.'
# column because there is tries in the score'
pd_no=pd_no.drop('UPDRS The Unified Parkinson’s Disease Rating Scale (UPDRS) score that isassigned to the subject by a physician via a medical examination todetermine the severity and progression of Parkinson’s disease.',axis=1)
pd_yes=pd_yes.drop('UPDRS The Unified Parkinson’s Disease Rating Scale (UPDRS) score that isassigned to the subject by a physician via a medical examination todetermine the severity and progression of Parkinson’s disease.',axis=1)


# In[47]:


# correlation function


def correlation(x,y):
#     Pearson's correlation coefficient
    corr_matrix = np.corrcoef(x, y)
    corr_coeff = corr_matrix[0, 1]
    print('Pearson correlation coefficient : ',corr_coeff)   
    
#     kendalltau correlatio coefficient
    acorr_coeff, p_value = kendalltau(x, y)
    print('kendalltau correlation coefficient : ',acorr_coeff) 
    
    
# spearmanr correlation coefficient
    bcorr_coeff,bn=spearmanr(x, y)
    print('spearmanr correlation coefficient : ',bcorr_coeff)
    
#     point biserial correlation coefficient
    point_biserial_corr, p_value = stats.pointbiserialr(x, y)
    print("point biserial correlation coefficient : ", point_biserial_corr)
    
#     taking avergae of correlations
    av=(float(corr_coeff)+float(acorr_coeff)+float(bcorr_coeff)+float(point_biserial_corr))/4
    
    return av
#     return [corr_coeff*100,acorr_coeff*100]


# In[49]:


di=dict()
for name_of_feature in pd_no:
    print(name_of_feature)
    not_pd=pd_no[name_of_feature].astype(float).tolist()
#     df['Numbers']
    yes_pd=pd_yes[name_of_feature].astype(float).tolist()
#     print(aaa)
    
    di[name_of_feature]=correlation(not_pd,yes_pd)
    print("\n")
    
#     df['Numbers'].astype(float).tolist()
    


# In[52]:


# ranking of salient variables (features) 
sorted_keys = sorted(di, key=lambda x: abs(di[x]), reverse=False)
print("Ranking of salient variables (features)\n")

for number,data in enumerate(sorted_keys):
      print(f"Rank {number+1} : {data}") 


