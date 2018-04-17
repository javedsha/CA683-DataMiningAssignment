#
#R Code for visualisation of the LDA topic modelling of the 20Newsgroup dataset.
#
#Reference code taken from: https://www.kaggle.com/solution/lda-visualization/code
#
#Install and load relevant packages
install.packages("readr")
install.packages("tm")
install.packages("lda")
install.packages("LDAvis")
install.packages("servr")
install.packages("quanteda")
install.packages("data.table")
install.packages("Matrix")
library(readr)
library(tm)
library(lda)
library(LDAvis)
library(servr)
library(quanteda)
library(data.table)
library(Matrix)

#set working directory
setwd("C:/Users/kshortall")

#read the input data
Input <- readLines("list_of_words-comma.txt")

#Pre-processing
Input <- gsub(",", " ", Input)  # remove commas
Input <- gsub("^[[:space:]]+", "", Input) #remove whitespace at beginning of documents
Input <- gsub("[[:space:]]+$", "", Input) #remove whitespace at end of documents

#Tokenize on space and output as a list
doc.list <- strsplit(Input, "[[:space:]]+") 
head(doc.list)

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)
head(term.table)
#Remove stop words
stopwords <- stopwords(language = "en", source = "smart")
stopwords
del <- names(term.table) %in% stopwords
term.table <- term.table[!del]
vocab <- names(term.table) #vocab is character vector containing the terms. Used in LDA model input.
head(vocab)

#putting the documents into a format required of the LDA package
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms) #a list of length equal to the number of documents.

# Compute some statistics related to the data set:
D1 <- length(documents)
D1 #Shows the number of documents in the corpus
W1 <- length(vocab)
W1 #Shows the number of terms in the vocabulary
doc.length <- sapply(documents, function(x) sum(x[2, ])) #a numeric vector with token counts for each document
N1 <- sum(doc.length)  
N1 #Shows total number of tokens in the corpus
term.frequency <- as.integer(term.table) #a numeric vector of observed term frequencies

# Markov-Chain Monte Carlo and model tuning parameters:
K1 <- 20 #Number of topics required.
G1 <- 1000 #Number of sweeps of Gibbs sampling over the entire corpus to make.
alpha1 <- 0.02 #Scalar value of the Dirichlet hyperparameter for topic proportions.
eta1 <- 0.02 #Scalar value of the Dirichlet hyperparameter for topic multinominals.

# Fit the model:
set.seed(1)
t_init <- Sys.time() 
fit1 <- lda.collapsed.gibbs.sampler(documents = documents, K = K1, vocab = vocab, 
                                    num.iterations = G1, alpha = alpha1, 
                                    eta = eta1, initial = NULL, burnin = 0,
                                    compute.log.likelihood = TRUE)
t_final <- Sys.time()
t_final - t_init #total time taken to fit the LDA model.  

### Visualizing the fitted model with LDAvis
theta1 <- t(apply(fit1$document_sums + alpha1, 2, function(x) x/sum(x)))
phi1 <- t(apply(t(fit1$topics) + eta1, 2, function(x) x/sum(x)))

results <- list(phi = phi1,
                 theta = theta1,
                 doc.length = doc.length,
                 vocab = vocab,
                 term.frequency = term.frequency)


# create the JSON object to display the visualisation:
json <- createJSON(phi = results$phi, 
                    theta = results$theta, 
                    doc.length = results$doc.length, 
                    vocab = results$vocab, 
                    term.frequency = results$term.frequency)

#Diplay the visualisation in a browser:
serVis(json, out.dir = './', open.browser = TRUE)