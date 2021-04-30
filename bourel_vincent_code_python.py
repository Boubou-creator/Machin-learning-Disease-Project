#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import svm


# In[3]:


import csv
with open('BC1.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        print(', '.join(row))


# In[13]:


import csv
x=[]
with open('BC1.csv','r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader :
        x.append(row[2])
print(x)


# In[ ]:




