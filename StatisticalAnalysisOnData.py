
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_20newsgroups


# In[2]:


train = fetch_20newsgroups(subset='train')


# In[17]:


print("\n".join(train.data[15].split("\n"))) #prints first line of the first data file


# In[19]:


# Extracting features from text files
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train.data)
X_train_counts.shape


# In[20]:


# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# In[ ]:


# Statistical Analysis


# In[66]:


test = fetch_20newsgroups(subset='test')


# In[78]:


full = train.data + test.data


# In[79]:


full_traget = list(train.target) + list(test.target)
len(full_traget)


# In[80]:


words_count = [];


# In[81]:


for i in range(0, len(full_traget)):
    words_count.append(len(full[i].split()))


# In[82]:


average_word_count = sum(words_count) / len(words_count)


# In[83]:


print(average_word_count)


# In[84]:


import statistics
statistics.median(words_count)


# In[85]:


max(words_count)


# In[86]:


min(words_count)


# In[87]:


from matplotlib import pyplot as plt


# In[88]:


import numpy as np


# In[89]:


plt.hist(words_count, bins=np.arange(14, 1000, 5))
plt.title('Newsgroup articles word counts')
plt.xlabel('words')
plt.ylabel('count')

plt.show()


# In[90]:


from collections import Counter


# In[91]:


c = Counter(full_traget)


# In[92]:


c.values()


# In[93]:


c.keys()

