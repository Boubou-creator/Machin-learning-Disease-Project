#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas
import numpy as np
import seaborn as sns
import matplotlib as plt
from sklearn.metrics import classification_report, confusion_matrix


# In[4]:


data = pandas.read_csv("ChronicDisease.csv")


# In[5]:


data.head()


# In[4]:


data.isna().mean()

On va supprimer les colonnes avec plus de 20% de valeurs manquantes
# In[5]:


a=data.isna().mean().tolist()
b=data.columns.tolist()
c=[]
for k in range(len(a)):
    if a[k] >= 0.2 :
        c.append(b[k])
print(c)


# In[6]:


data2=data.drop(columns=c)
data2

l'id fait office de doublure de la clef primaire on peut donc la supprmer
# In[7]:


data3=data2.drop(columns='Id')

on va isolé et transformé en catégorie les colonnes de types object
# In[8]:


catego=data3.select_dtypes(include='object').astype('category')


# In[9]:


catego=pandas.get_dummies(catego, drop_first=True)

on va réunier les catégories avec les donnée numériques 
# In[10]:


data3=data3.select_dtypes(exclude='object')
data4=data3.join(catego)
data4


# In[11]:


corr=data4.corr()
hm=sns.heatmap(corr)
hm

on va identifier puis retirer les colonnes qui on moins de 50% de corrélation avec la classe
# In[12]:


a=corr.index.tolist()
b=corr['Class_notckd'].tolist()
c=[]
for k in range(len(a)):
    if abs(b[k])<=0.5:
        c.append(a[k])
print(c)


# In[13]:


data5=data4.drop(columns=c)
data5

On va remplier les données manquantes avec la moyenne de la colonnes (même si cela n'a pas de sens pour les catégories)
# In[14]:


a=data5.mean().tolist()
b=data5.columns.tolist()
f={}
for k in range(len(a)):
        f.update({b[k]: a[k]})
print(f)


# In[15]:


data6 = data5.fillna(value=f)


# # Fonction preprocessing

# In[16]:


def preprocessing(data):
    a=data.isna().mean().tolist()
    b=data.columns.tolist()
    c=[]
    for k in range(len(a)):
        if a[k] >= 0.2 :
            c.append(b[k])
    data2=data.drop(columns=c)
    data3=data2.drop(columns='Id')

    catego=data3.select_dtypes(include='object').astype('category')
    catego=pandas.get_dummies(catego, drop_first=True)
         
    data3=data3.select_dtypes(exclude='object')
    data4=data3.join(catego)
    corr=data4.corr()           
    d=corr.index.tolist()
    e=corr['Class_notckd'].tolist()
    f=[]
    for k in range(len(d)):
        if abs(e[k])<=0.5:
            f.append(d[k])
    data5=data4.drop(columns=f)

    g=data5.mean().tolist()
    h=data5.columns.tolist()
    i={}
    for k in range(len(h)):
            i.update({h[k]: g[k]})
    data6 = data5.fillna(value=i)
    return(data6)


# In[17]:


preprocessing(data)


# # ML ALGORITHM 

# In[18]:


X= data6.drop(columns='Class_notckd')
Y= data6['Class_notckd']

on génère des sets d'entrainement et de testes pour les modèles
# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30)

On va utiliser un préscaler pour améliorer le set d'entrainement
# In[20]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

On retrouve ici les paramètre sélectionné (j'ai supprimé certain parmètre comme le kernal polynomial du svm en raison du temps de calcul)
# In[21]:


from sklearn.model_selection import GridSearchCV
param_svc ={           'C':[ 1, 10, 50, 100],           'kernel':['linear',  'rbf'],            'gamma':['scale', 'auto'],           }

param_dt ={           'min_samples_split':[2,3,5,10,20],           'max_depth':[1, 5, 10, 20, 100],           }
param_knn ={             'algorithm': ['auto'],
             'metric': ['manhattan'],
             'n_neighbors': [3, 5, 7,8,9],
             'weights': ['uniform', 'distance']\
            }


# # SVM
on utilise un optimeur hyperparamétrique
# In[22]:


from sklearn.svm import SVC
classifier = SVC()
grid = GridSearchCV(classifier, param_grid=param_svc,
                    cv=50)

grid_results = grid.fit(X,Y)

print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))

on entrainne le logiciel avec les meilleurs paramètres
# In[23]:


SVM = SVC(**grid_results.best_params_)
SVM.fit(X_train,y_train)


# In[24]:


y_pred_svm = SVM.predict(X_test)

on évalue les performance 
# In[25]:


print(confusion_matrix(y_test,y_pred_svm))
print(classification_report(y_test,y_pred_svm))


# # KNN

# In[26]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
grid = GridSearchCV(classifier, param_grid=param_knn,
                    cv=50)

grid_results = grid.fit(X, Y)

print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))


# In[27]:


KNN = KNeighborsClassifier(**grid_results.best_params_)
KNN.fit(X_train,y_train)


# In[28]:


y_pred_knn = KNN.predict(X_test)


# In[29]:


print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))


# # Decision Tree

# In[30]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
grid = GridSearchCV(classifier, param_grid=param_dt,
                    cv=50)

grid_results = grid.fit(X, Y)

print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))


# In[31]:


Decision_tree = DecisionTreeClassifier(**grid_results.best_params_)
Decision_tree.fit(X_train,y_train)


# In[32]:


y_pred = Decision_tree.predict(X_test)


# In[33]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# #  Production model
Les meilleurs scores dans un temps de calcul très cour, sont obtenues avec les decision tree classifier DecisionTreeClassifier(max_depth=100, min_samples_split=3)
# In[34]:


production_model=DecisionTreeClassifier(max_depth=100, min_samples_split=3)

