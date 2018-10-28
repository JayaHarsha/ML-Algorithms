
# coding: utf-8

# In[1]:


import numpy as np
get_ipython().magic('matplotlib inline')
import pandas as pd
from scipy.io import arff
from matplotlib import pyplot as plt
from scipy.spatial import distance 
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

#loading the data using arff
data1 = arff.loadarff("C:/Users/Harsha/Desktop/machine learning/EEG Eye State.arff.txt")
#Loading the data using Pandas
np.random.seed(35)

dataframe = pd.DataFrame(data1[0])
dataframe['eyeDetection'] = dataframe['eyeDetection'].astype('int64')
maindf, testdf = train_test_split(dataframe, test_size=0.2)

#del maindf['eyeDetection']
#maindf.info()

for f in list(maindf.columns.values):
    if f != 'eyeDetection':
        maindf[f] = (maindf[f] - maindf[f].mean() )/maindf[f].std()
    else: 
        print(True)
    
#print(maindf)
df, valdf = train_test_split(maindf, test_size = 0.25)
y1 = df["eyeDetection"]
del df["eyeDetection"]
#print(y1)
y2 = valdf["eyeDetection"]
del valdf["eyeDetection"]


# In[3]:


for f in list(maindf.columns.values):
    maindf[f] = (maindf[f] - maindf[f].mean() )/maindf[f].std()
    
#print(maindf)
df, valdf = train_test_split(maindf, test_size = 0.25)


# In[4]:




#Euclidean Distance to compute the distance
def eucdist(x,y):
    return np.linalg.norm(np.asarray(x)-np.asarray(y))
     

    


# In[5]:


a =[1,2,3,4,4]
b = [3,4,5,6,7]
eucdist(a,b)


# In[56]:


#The main clustering function
def Cluster(k, df):
    lenofdf = len(df)
    centre = {}
    #dist = []
    #this dictionary consists of the distances of different nodes to their centroids
    for i in range(lenofdf):
        classification = {} 
    
    #generating a random number in the length of the dataframe taking that as the centroid 
    for i in range(k):
        classification[i] = []
        centre[i] = df.iloc[np.random.randint(0,lenofdf-1)]
        
    count = 0
    #for each Instance in the dataframe calculating the distance to the current centroids
    for index, d in df.iterrows():
        for cent in centre:
            dist = []
            count += 1
            #appendidng the distance to a list
            #print("df",type(d))
            #print("center",centre[cent])
            dist.append(eucdist(d, centre[cent]))
            #classifying the instance into the type with minimum distance
            #print(dist)
        classify = dist.index(min(dist))
        #print(dist)
        classification[classify].append(d)
    #print(classification[classify])
    print(count)
    
    #defining a dictionary for all the centroids 
    oldcentre = dict(centre)
    
    ##calucaling the new position of centroid once every instance is classified
    for classify in classification:
        print(classify)
        centre[classify] = np.average(classification[classify], axis = 0)
     
    
    
    #declared a  variable for stopping condition    
    #Giving a condition to stop if there is no difference between previous and new centroids then stop
    for c in centre:
        precentre = oldcentre[c]
        newcentre = centre[c]
        print('diff ', eucdist(newcentre, precentre))
        if np.sum(eucdist(newcentre,precentre)>0.0001):
            pass
        else:
            break
            
     
    return centre
        


# In[57]:


## this function returns the predicted values i.e. the closest cluster by calculating the distance
def predict(df, centre, y):
    dist1 = []
    for cent in centre:
        dist1.append(eucdist(df,centre[cent]))
    classify = dist1.index((min(dist1)))
    print(dist1)
    print(classify)
    return classify, y.iloc[classify]


# In[58]:


a = Cluster(3, df)


# In[52]:


##for the training set repeating it for 30 times
k = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
final_prob = {}
ent = []
for z in k:
    print(z) 
    for m in range(30):
        x = []
        print(z)
        clf =  Cluster(n_clusters=z, df )
        final_prob[z] = []
        
        mydict = {i: np.where(clf.labels_ == i)[0] for i in range(clf.n_clusters)}

        # Transform this dictionary into list (if you need a list as result)
        dictlist = []
        for key, value in mydict.items():
            temp = [key,value]
            dictlist.append(temp)
        prob = []
        pred = []
        for c in range(clf.n_clusters):
            for i in dictlist[c][1]:
                #print(i)
                pred.append(y1.iloc[i])
            #print(pred)
            prob.append(pred.count(1)/len(pred))
        x.append(log_loss( y1, predict(df)))
        print(x)
    ent.append(np.sum(x)/len(x))
    


# In[ ]:


##for the validation repeating for 30 times
k = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
final_prob = {}
ent = []
for z in k:
    print(z) 
    for m in range(30):
        x = []
        print(z)
        clf =  Cluster(n_clusters=z, df )
        final_prob[z] = []
        
        mydict = {i: np.where(labels_comparision == i)[0] for i in range(z)}

        # Transform this dictionary into list (if you need a list as result)
        dictlist = []
        for key, value in mydict.items():
            temp = [key,value]
            dictlist.append(temp)
        prob = []
        pred = []
        for c in range(z):
            for i in dictlist[c][1]:
                #print(i)
                pred.append(y1.iloc[i])
            #print(pred)
            prob.append(pred.count(1)/len(pred))
        x.append(log_loss( y1, predict(df)))
        print(x)
    ent.append(np.sum(x)/len(x))


# In[ ]:


xfin = []
clf =  Cluster(n_clusters=5, df)
final_prob[z] = []

mydict = {i: np.where(labels_comparision == i)[0] for i in range(k)}

# Transform this dictionary into list (if you need a list as result)
dictlist = []
for key, value in mydict.items():
    temp = [key,value]
    dictlist.append(temp)
prob = []
pred = []
for c in range(n_clusters):
    for i in dictlist[c][1]:
        #print(i)
        pred.append(yfinal.iloc[i])
    #print(pred)
    prob.append(pred.count(1)/len(pred))
xfin.append(log_loss( yfinal, predict(testdf)))
print(xfin)

