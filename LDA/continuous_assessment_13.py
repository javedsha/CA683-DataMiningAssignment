"""
Author: Ken Brennock
Comment:10/03/18 - Inital work on analysing the data.
       :27/03/18 - Found out that there are duplicate file names in the different source folders, had to put in
	               a method to ensure that all file names are unique. Added a number at the end of each file name.
	   :01/04/18 - Version 05 - looking to do the LDA
	   :02/04/18 - Version 07 - removed all words that only appear once, on the recommendation of the website
	                          - https://radimrehurek.com/gensim/tut1.html
	   :12/04/18 - Version 08 to 11 - added in an array for out own stop words, prints out the topics and related key words
		   
"""

#import csv #imports the csv module
#import sys #import  the sys module
#import re  #import the reg. expression module
import os
#from operator import itemgetter #used to get information from lists
from time import gmtime, strftime
from pandas import DataFrame
import numpy as np
from nltk.corpus import stopwords #stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import string
from string import digits
from gensim import corpora
from operator import itemgetter
from pathlib import Path
#The appraoch to LDA supported by https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/
#Beginners Guide to Topic Modeling in Python,  Shivam Bansal , August 24, 2016
import gensim
import csv

print('Start time of the File processing', (strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))
NEWLINE = '\n'
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
remove_digits = str.maketrans('', '', digits)
N = 50
number_of_topics = 20
number_of_passes = 200
words = {}
clean_documents = []
my_stopwords = ['subject', 'line', 'organization', 're', 'one', 'b', 'l', 'r', 'f', 'k', 'g', 'h', 'w', 'q', 'm', 'c', 'v', 'p', 'e', 's', 'n', 'a', 'x', 'u', 'i', 'writes']

folder = '20news-18828'
folder_save = 'Clean_Full'

#folder = 'testing'
#folder_save = 'Clean_Testing'


# Beginners Guide to Topic Modeling in Python -  Shivam Bansal - 
# URL - https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/
# First line removes stop words, second line remove all the numbers, third remove punuaction, the forth removes any
# none English words and the final line stems the remaining text.
def clean(text):
	stop_free = ' '.join([i for i in text.lower().split() if i not in stop])
	num_free = stop_free.translate(remove_digits)
	punc_free = ''.join(ch for ch in num_free if ch not in exclude)
	english = ' '.join(eng_word for eng_word in punc_free.split() if wordnet.synsets(eng_word))
	stemmed = ' '.join(lemma.lemmatize(word) for word in english.split())
	stemmed = ' '.join([j for j in stemmed.split() if j not in my_stopwords])
	return stemmed

# Zac Stewart at URL - http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html
# This functions has been heavely modified from Zac Stewarts.
# This recrussively reads the files from a path and cleans each file as it goes 
def read_files(path):
	print('Reading all the files')
	#This reads all the files in a paths, including all sub directories
	#Then for each file it reads each line and put those lines into a array called lines
	#Once the file has been completely read the newlines are removed and the result is put into and array called content
	#The contents are then cleaned - calling the function 'clean'
	for root, dirs, files in os.walk(path, topdown=False):
		for name in files:
			full_file_name = (os.path.join(root, name))
			if os.path.isfile(full_file_name):
				lines = []
				#print('Full File Name ---> ',full_file_name)
				#print('File Name ---> ',root)
				f = open(full_file_name, 'r')
				for line in f:
					lines.append(line)
				f.close()
				content = NEWLINE.join(lines)
				#print('Number of words before the clean ---> ', len(content.split(' ')))
				#print('Content Before Clean ---> ', content)
				content = clean(content)
				#print('Number of words after the clean ---> ', len(content.split(' ')))
				#print('Content After Clean ---> ', content)
				#content = [clean(doc).split() for doc in content]
			yield name, content

print('Before read_files')
rows = []
index = []

file_count = 1
file_names_store = []


# For every file found recursive desent though 'folder', the words are counted and put into a variable of type dictionary
for file_name, text in read_files(folder):
	if (len(text) < 100000) and (len(text) > 0):
		new_file = folder_save+'\\'+str(file_count)+'_'+file_name
		f = open(new_file, 'w')
		f.write(text)
		f.close()
		#Put the clean documents into and array
		#Storing the file names in an array to be used later, to provide the names against the topics.
		#print('Adding new file to list',new_file)
		file_names_store.append(str(file_count)+'_'+file_name)
		clean_documents.append(text)
		#print('TEXT --->',text)
		#topic_words.writerow(text.split(' '))
		for word in text.split(' '):
			words[word] = words.get(word, 0) + 1
		file_count+= 1

	
print('Total Unique Words size ----->', len(words))
print('Read all the files', (strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))
#print('There are the following unique words ----->',len(words))
#print()

#Take the clean documents and put all the words into an array.
#These words are not unique and it would be expected that a lot of them would be repeated
list_of_words = [document.split() for document in clean_documents]
print('Created the list of words', (strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))
#print('List of works, first ---->',list_of_words)
#print()

for text in list_of_words:
	#print('First Index --->',list_of_words.index(text))
	first_index = list_of_words.index(text)
	for word in text:
		second_index = list_of_words[first_index].index(word)
		#Remove the words are unique
		if words[word] <= 1:
			#print('Second Index --->',second_index)
			#print('Single count word ---->',word)
			list_of_words[first_index].pop(second_index)

with open('file_length.csv','w') as file:
	for document_temp in list_of_words:
		file.write(str(len(document_temp)))
		file.write('\n')

#print('List of works, second ---->',list_of_words)
#print()
f1 = open('list_of_words-comma.txt', 'w')
f2 = open('list_of_words-space.txt', 'w')
for documents in list_of_words:
	f1.write(','.join(documents)+'\n')
	f2.write(' '.join(documents)+'\n')

f1.close()
f2.close()

dictionary = corpora.Dictionary(list_of_words)
print('Total Dictionary size ----->', len(dictionary))
print('Created dictionary', (strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))
#print('Dictionary ------->', dictionary)
print()
#print(dictionary.token2id)
print()

doc_term_matrix = [dictionary.doc2bow(doc) for doc in list_of_words]
print('created the vector', (strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))
#print(doc_term_matrix)

Lda = gensim.models.ldamodel.LdaModel
print('Start the modelling', (strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))
ldamodel = Lda(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary, passes=number_of_passes)
print('Finish the modelling', (strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))

#for top in ldamodel.print_topics():
#	print(top)
#	print()
	
with open('topic_words.csv','w') as file:
    for top in ldamodel.print_topics():
        file.write(str(top))
        file.write('\n')

#Get the documents that went to make up the corpus
test_corpus = ldamodel[doc_term_matrix]


#for doc in test_corpus:
#	print('Print --->',doc)

	
scores = []
count = -1
groups = {}
keys = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
groups = dict(zip(keys, [None]*len(keys)))

for key in groups:
	groups[key] = []

f3 = open('Documents_and_Topics.txt', 'w')
	
#https://stackoverflow.com/questions/20984841/topic-distribution-how-do-we-see-which-document-belong-to-which-topic-after-doi
print('Start putting the documents in their topics', (strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))
for doc in test_corpus:
	#print()
	count +=1
	biggest_score = 0
	#print('******DOC --->',file_names_store[count])
	for topic in doc:
		score_new = topic[1]
		topic_name = topic[0]
		#print('Length of topic',len(topic))
		#print(file_names_store[count]+' for topic ',topic_name,' is ',score_new)
		if score_new > biggest_score:
			biggest_score = score_new
			biggest_topic = topic_name
	#print(file_names_store[count]+' larges score ',' is ',biggest_score, biggest_topic)
	line_to_write = file_names_store[count]+'--->'+str(doc)
	f3.write(''.join(line_to_write)+'\n')
	groups[biggest_topic].append(file_names_store[count])

f3.close()
#print(groups.items())
print('Finshed putting the documents in their topics', (strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))

total_length = 0
for key in groups:
	#print('GROUP ',key,' --->',groups[key],len(groups[key]))
	total_length += len(groups[key])
	topic_file_name = 'Topics\\'+str(key)+'_'+str(len(groups[key]))+'.txt'
	with open(topic_file_name, 'w') as file:
		file.write(str(groups[key]))
	print()
		
#print('Total Length',total_length)
			
	
print('The top ',N,' words')
top_words = sorted(words.items(), key=itemgetter(1), reverse=True) [:N]
	
for word, frequency in top_words:
	print('%s: %d' % (word, frequency))

print('After read_files')
print('End time of the File processing', (strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))
