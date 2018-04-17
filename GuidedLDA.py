
# coding: utf-8

# In[51]:


import numpy as np


# In[52]:


import guidedlda


# In[83]:


from sklearn.datasets import fetch_20newsgroups
data1 = fetch_20newsgroups(shuffle=True, subset='train', remove=('From', 'headers', 'footers'))
data2 = fetch_20newsgroups(shuffle=True, subset='test')


# In[90]:


fd = data1.data + data2.data


# In[93]:


fd[1]


# In[97]:


import re
fd_re = []
# remove digits
for i in range(0, len(fd)):
    fd_re.append( re.sub("^\d+\s|\s\d+\s|\s\d+$", "", fd[i]))


# In[99]:


fd_re[1]


# In[109]:


import nltk
nltk.download()


# In[113]:


stopWords = frozenset(nltk.corpus.stopwords.words('english'))
stopWords.union(frozenset(['.net', '.com', '.edu', '.org']))


# In[114]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words=stopWords)
data_tokens = count_vect.fit_transform(fd_re)
data_tokens.shape


# In[115]:


# create seed topic list
seed_topic_list = [['windows', 'dos', 'os', 'ms', 'microsoft'],
                  ['god', 'jesus', 'bible', 'christ', 'believe', 'faith'],
                  ['key', 'chip', 'encryption', 'clipper', 'keys', 'escrow']]


# In[116]:


# convert the above seed topic list, to vector
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[count_vect.vocabulary_[word]] = t_id


# In[118]:


model = guidedlda.GuidedLDA(n_topics=20, n_iter=200, refresh=20, random_state=7)


# In[119]:


model.fit(data_tokens, seed_topics=seed_topics, seed_confidence=0.45)


# In[120]:


# Display the top 10 words for each topics
n_top_words = 10
topic_word = model.topic_word_
#print(topic_word)
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(count_vect.get_feature_names())[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

