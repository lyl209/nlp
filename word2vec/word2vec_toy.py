
# coding: utf-8

# In[1]:

from gensim.models import word2vec
import random
import numpy as np


# In[2]:

cbow_model = word2vec.Word2Vec.load_word2vec_format('f:/Data/xxx.bin', binary=True)  # C binary format


# In[3]:

print cbow_model.most_similar('crap')


# In[4]:

print cbow_model.most_similar('unacceptable')


# In[5]:

print cbow_model.most_similar('hello')


# In[6]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots


# In[7]:

visualizeWords = ['Oh','oh','Oh,','haha','crap','hi','working?','hours','awful','spent','ridiculous', 'HI','yo','Hello','hi','apology', 'good', 'Enjoy', 'nobody', 'idiot', 'escalate', 'hello','great','unacceptable']
visualizeList = []
for word in visualizeWords:
    row = cbow_model[word]
    visualizeList.append(row)

visualizeVecs = np.vstack(visualizeList)

temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeWords) * temp.T.dot(temp)
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2]) 

for i in xrange(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i], bbox=dict(facecolor='green', alpha=0.1))
    
plt.xlim((np.min(coord[:,0]) - 0.1, np.max(coord[:,0]) + 0.1))
plt.ylim((np.min(coord[:,1]) - 0.1, np.max(coord[:,1]) + 0.1))


# In[ ]:



