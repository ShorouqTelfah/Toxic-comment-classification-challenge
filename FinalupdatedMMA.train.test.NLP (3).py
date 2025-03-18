#!/usr/bin/env python
# coding: utf-8

# # Perform detailed data analysis of the dataset 

# In[1]:


# load the train file 
import pandas as pd


# In[2]:


df = pd.read_excel(r"C:\Users\shoro\OneDrive\Desktop\train.xlsx")


# In[3]:


# overview the dataset 
df


# In[4]:


# number of rows and columns 
df.shape


# In[5]:


# the name of features

df.columns


# In[7]:


# inspect the first five rows 
print (df.head())


# In[8]:


# I will check for the missing value 
print (df.isnull().sum())


# In[9]:


#quick over view
# sentences marked with a 1 per class
#I will check the data set if balanced or not 
# will check the class distribution 
# then will go with the details 
df[['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']].sum()


# In[10]:


# I will do cleaning for the dataset 

import re

# definea function to clean the text.

def clean_text(text):
    # lowercase the text     
    text = text.lower()

    # remove URLs if present    
    text = re.sub(r'http\S+', '', text)

    # keep only letters      
    text = re.sub(r'[^a-zA-Z\s]', '', text)

   
    # treat spaces      
    text = re.sub(r'\s+', ' ', text).strip()

    return text 
    


# In[11]:


# apply the cleaning function to 'comment_text'   
df['comment_text'] = df['comment_text'].apply(clean_text)


# In[12]:


# check the previous step
print(df[['comment_text']].head())


# In[13]:


df.shape


# # Number of sentence per class

# In[14]:


# I WILL CHECK THE DATA IF BALANCED OR NOT 

total_sentence_class = df[['toxic', 'severe_toxic', 'obscene', 'threat',
                 'insult', 'identity_hate']].sum()


# In[15]:


# Print the total number of sentences for each class
print("Total number of sentences per class:")
for class_name, count in total_sentence_class.items():
    print(f"{class_name}: {count}")


# In[149]:


# i will import pyplot library

import matplotlib.pyplot as plt

#create a Bar plot 
total_sentence_class.plot(kind='bar', color='lightblue', edgecolor='black')

# name the plot,x and y axis

plt.title('Toxicity label destribution across classes ')
plt.xlabel('the types of toxicity')
plt.xticks(rotation=45, ha='right')
plt.ylabel(' TOTAL Count of Instances')

# show the plot 

plt.tight_layout()
plt.show()


# # Number of token per class

# In[34]:


# now I will calculate the number of words in each class
#new df will be created whith only comment_text 

# first I will define a count_tokes function 
def count_tokens(text):
    return len(str(text).split())

# Creating a new dataframe with token counts and the class labels
class_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df_total_tokens = df[['comment_text'] + class_columns].copy()

# for each sentence we will add column to show the number of token 

df_total_tokens['token_count'] = df_total_tokens['comment_text'].apply(lambda x: count_tokens(str(x)))

# then I will initialize a dic as stor for the number of the tokens 

total_token_to_class = {}

# go over each class then calculate 

for class_name in class_columns:
    # filter recordes where the class labeled 1
    
    class_sentences = df_total_tokens[df_total_tokens[class_name] == 1]
    
    # calculate the total
    
    total_tokens = class_sentences['token_count'].sum()
    
    #put in the dictionary 
    total_token_to_class[class_name] = total_tokens

# shoe the result 
for class_name, total_tokens in total_token_to_class.items():
    print(f" the total number of token in  '{class_name}': {total_tokens}")


# In[23]:


# I will import the needed library 

import matplotlib.pyplot as plt
import seaborn as sns

# I will generate visualize data analysis bar plot 

plt.figure(figsize=(10, 6))# the size of the figure

#creating the bar

sns.barplot(x=list(total_token_to_class.keys()), y=list(total_token_to_class.values()), palette='Blues')

# finishing the plot 

plt.xticks(rotation=45, ha='right')  #for better apearance 
plt.xlabel("the types of toxicity ")
plt.ylabel("the total number of tokens")
plt.title("Ttotal number of tokens per each types of toxicity")

# disply 
plt.tight_layout()
plt.show()


# # word frequency analysis
# to analyz the most frequent word used in comments for each toxicity class

# In[24]:


# import needed libraries 

from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

# built the needed function 

def word_frequency_by_class(df, class_columns, top_n=10):
    
    # to Initialize CountVectorizer to tokenize words
    vectorizer = CountVectorizer(stop_words='english')
    
    # creat a dictionary to hold the word frequencies per class 
    
    class_word_frequencies = {}

    for class_name in class_columns:
        # Filter sentences where the class is 1 
        class_sentences = df[df[class_name] == 1]['comment_text']
        
        # Apply vectorizer to the filtered sentence 
        
        word_matrix = vectorizer.fit_transform(class_sentences)
        
        #get the word frequency count 
        
        word_counts = word_matrix.sum(axis=0).A1
        words = vectorizer.get_feature_names_out()
        
        
        word_freq = dict(zip(words, word_counts))
        
        # sort words by frequency and take the top-n most common 
        
        sorted_word_freq = Counter(word_freq).most_common(top_n)
        
        # Store the top_n frequent words for each class
        class_word_frequencies[class_name] = sorted_word_freq

    return class_word_frequencies

# get top 10 word for each class 

top_words_by_class = word_frequency_by_class(df, class_columns, top_n=10)

# plotting the result 

plt.figure(figsize=(15, 10))
for i, (class_name, word_freq) in enumerate(top_words_by_class.items()):
    words, counts = zip(*word_freq)
    
    # separate plot for each class 
    
    plt.subplot(2, 3, i + 1)
    sns.barplot(x=list(counts), y=list(words), color='b')
    plt.title(f"Top Words in the Class: {class_name}")
    plt.xlabel("Frequency of the words")
    plt.ylabel("the Word")

plt.tight_layout()
plt.show()


# In[25]:


# show the top 10 words with frequency in each class 

for class_name, word_freq in top_words_by_class.items():
    print(f"The most frequent words for each class '{class_name}':")
    for word, count in word_freq:
        print(f"  token : {word}, total frequency: {count}")
    print("\n")


# In[ ]:





# In[ ]:





# # Feature extraction TF-IDF Vectorizer

# In[26]:


#transform the'comment_text' column into term document matrix 
# import the TfidfVectorizer from sklearn

from sklearn.feature_extraction.text import TfidfVectorizer

# start initialize the TfidfVectorizer
# stop_words='english': filter only english words

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')# limit the number of parameter 

# then fit and transform the 'comment_text'

X_tfidf = tfidf.fit_transform(df['comment_text'])

print(X_tfidf.shape)


# In[27]:


#View the Feature Names:
feature_names = tfidf.get_feature_names_out()
print(feature_names[:10])  # print the first 10 features   


# In[28]:


# view document's TF-ID Vector:

first_document_tfidf = X_tfidf[0].toarray()
print(first_document_tfidf)


# In[29]:


# inspect Non-Zero TF-IDF Values 
# to now which terms in the document are contributed to the overall TF-IDF score 

import numpy as np
# change the format 

rows, cols = X_tfidf.nonzero()

# get the corresponding words and TF-ID value 

words = np.array(feature_names)[cols]
values = X_tfidf.data

# first 10 non-zero values 
for word, value in zip(words[:10], values[:10]):
    print(f"Word: {word}, TF-IDF: {value}")


# # Machine learning algorithms and their performance 

# In[31]:


# Import necessary libraries
#for spliting the data 
#for check the performance 
#for dealing with the imbalance 

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler


# In[32]:


X = df['comment_text'] # defining the feature 
y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] # defining the target 

# display the y shape 
print(y.shape)  


# # Logistic Regression 

# In[33]:


from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report

# split the data 80% training and 20% testing 
# random_state=42 :to ensure same split all the time 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# alredy done first: TF-IDF Vectorizer to transform the text data into feature vectors
# max_features=5000 :limit the number of word
#stop_words='english': remove unnessesary words 
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Fit and transform the training and test data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Check the shape of :

print("Train TF-IDF Shape:", X_train_tfidf.shape)
print("Test TF-IDF Shape:", X_test_tfidf.shape)


# In[160]:


# initiate the logistic regression model
# this parameter shows the maximum number of iteration that will be run 1000 
#class_weight='balanced' will handel the imbalance in the dataset 

logreg = LogisticRegression(max_iter=1000, class_weight='balanced')  
                                                        
# use MultiOutputClassifier as each comment can be assigned multiple labels 
#n_jobs=-1 : speed up the training process 
logreg_model = MultiOutputClassifier(logreg, n_jobs=-1)

#train the model 
logreg_model.fit(X_train_tfidf, y_train)

# make predivtion 
y_pred_logreg = logreg_model.predict(X_test_tfidf)

# evaluate the performance 
#classification_report :this function generate performance report 
print("LogReg. Performance Report:")
print(classification_report(y_test, y_pred_logreg, target_names=y.columns))

# chech the accurecy
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"LogReg. Accuracy: {accuracy_logreg:.4f}")


# In[56]:


# import needed library 
from sklearn.metrics import roc_auc_score


# In[30]:


# AUC for Logistic regression 
auc_LogReg = roc_auc_score(y_test, y_pred_logreg, average='macro', multi_class='ovr')
print(f"LogReg. AUC: {auc_LogReg:.4f}")


# In[ ]:





# # Naïve Bayes 

# In[161]:


from sklearn.naive_bayes import MultinomialNB


# In[162]:


# initialize 
nb = MultinomialNB()


# In[163]:


#  again Wrap it in MultiOutputClassifier to handle multi-label classification
nb_model = MultiOutputClassifier(nb, n_jobs=-1)


# In[165]:


# Train the model
nb_model.fit(X_train_tfidf, y_train)


# In[166]:


# make prediction 
y_pred_nb = nb_model.predict(X_test_tfidf)


# In[167]:


# evaluate the performance 
print("NB Performance Report:")
print(classification_report(y_test, y_pred_nb, target_names=y.columns))


# In[168]:


# check the accuracy 
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"NB Accuracy: {accuracy_nb:.4f}")


# In[169]:


# calculate AUC 
auc_nb = roc_auc_score(y_test, y_pred_nb, average='macro', multi_class='ovr') 
#for multiclass classifier
print(f"NB AUC: {auc_nb:.4f}")


# In[ ]:





# # Random Forest

# In[170]:


#I WILL IMPORT RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier


# In[171]:


# initiate the model and define the rf as avariable
#n_estimators=100 : number of random trees in the forest 
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)


# In[172]:


# Wrap it in MultiOutputClassifier to handle multi-label classification
rf_model = MultiOutputClassifier(rf, n_jobs=-1)


# In[173]:


# i will check the shape of the training data 
print(X_train_tfidf.shape)
print(y_train.shape)


# In[174]:


# Train the model on the training data
rf_model.fit(X_train_tfidf, y_train)


# In[175]:


y_pred_rf = rf_model.predict(X_test_tfidf)


# In[176]:


# evaluate the performance 
print("RF performance Report:")
print(classification_report(y_test, y_pred_rf, target_names=y.columns))

# the Accuracy
accuracy_RF = accuracy_score(y_test, y_pred_rf)
print(f"RF Accuracy: {accuracy_RF:.4f}")


# In[177]:


# Calculate AUC 
auc_RF = roc_auc_score(y_test, y_pred_rf, average='macro')

# Print the AUC for Random Forest model
print(f"RF Macro-Averaged AUC: {auc_RF:.4f}")


# In[ ]:





# # Changing Feature extraction method-Word Embeddings

# In[45]:


# import needed library 
import gensim


# In[46]:


# Load pre-trained Google News embeddings (this file should be downloaded and available)
embedding_current_path = r'C:\Users\shoro\Downloads\GoogleNews-vectors-negative300.bin.gz'

# generate function to load word vectors from WordsVec format

vector_space  = gensim.models.KeyedVectors.load_word2vec_format(embedding_current_path, binary=True)


# In[47]:


# I will prepare a function to convert the sentence into numirical vector
def sentence_to_vector(sentence, vector_space, vector_size=300):
    # first split sentence to words 
    words = sentence.split()
    # then we will filter the words 
    existent_words = [word for word in words if word in vector_space]
    
    if existent_words:
        
        # if it is available compute the ava. word embedding
        
        word_embeddings = np.array([vector_space[word] for word in existent_words])
        sentence_vector = word_embeddings.mean(axis=0)
    else:
        # if it is not available show zero
        sentence_vector = np.zeros(vector_size)
    
    return sentence_vector # will compute the sentence vector


# In[48]:


#  prepare all the data 
# reload the feature x and the target y

X = df['comment_text']
y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

# will call each comment in x and processed by the previously built function sentence_to_vector
X_embeddings = np.array([sentence_to_vector(comment, vector_space) for comment in X])

# prepare the data by spliting it 80 to 20% training to testing 
X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=42)


# In[49]:


# Initialize models for MultiOutput Classification
# MultiOutputClassifier APPROCH WILL BE USE 
# EACH COMMENT COULD BE CLASSIFIED TO MORE THAT ONE LABEL
# lR 
logreg = LogisticRegression(max_iter=1000)
logreg_model = MultiOutputClassifier(logreg, n_jobs=-1)


# In[50]:


# NAIVE BAYES
nb = MultinomialNB()
nb_model = MultiOutputClassifier(nb, n_jobs=-1)


# In[51]:


# RANDOM FOREST
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model = MultiOutputClassifier(rf, n_jobs=-1)


# # Train and Evaluate Logistic Regression

# In[57]:


# first we will train LR model
logreg_model.fit(X_train, y_train)

# then we will do prediction 
y_pred_logreg = logreg_model.predict(X_test)

# now i will assess the performance 

print("LR  PERFORMANCE Report:")
print(classification_report(y_test, y_pred_logreg, target_names=y.columns))
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
auc_LogReg = roc_auc_score(y_test, y_pred_logreg, average='macro', multi_class='ovr')

print(f"LR Accuracy: {accuracy_logreg:.4f}")
print(f"LR AUC: {auc_LogReg:.4f}")


# # Train and Evaluate Naive Bayes

# In[58]:


# import the nessessary library 
import numpy as np

# i will check for negative value as i recieved an error 

if np.any(X_train < 0):
    print("yes there are negative value in the training dataset")
else:
    print("no negative value ")


# In[59]:


# to train NB i will replace the negative value with 0
X_train[X_train < 0] = 0


# In[60]:


# first train the model

nb_model.fit(X_train, y_train)

# then I will proceed in the prediction 

y_pred_nb = nb_model.predict(X_test)

# finally I will assess the performance of the model 

print("NB Performance Report:")
print(classification_report(y_test, y_pred_nb, target_names=y.columns))

accuracy_nb = accuracy_score(y_test, y_pred_nb)
auc_nb = roc_auc_score(y_test, y_pred_nb, average='macro', multi_class='ovr')# FOR MULTI LABEL CLASSIFIER
print(f"NB Accuracy: {accuracy_nb:.4f}")
print(f"NB AUC: {auc_nb:.4f}")


# In[ ]:





# In[ ]:





# # Train and Evaluate Random Forest

# In[61]:


#FIRST WILL FIT THE DATA 
rf_model.fit(X_train, y_train)

#THEN WE WILL MAKE PREDICTION 
y_pred_rf = rf_model.predict(X_test)

# FINALLY WE WILL ASSESS THE PERFORMANCE 
print("RF performance Report:")
print(classification_report(y_test, y_pred_rf, target_names=y.columns))

accuracy_RF = accuracy_score(y_test, y_pred_rf)
auc_RF = roc_auc_score(y_test, y_pred_rf, average='macro', multi_class='ovr') # FOR MULTI LABEL CLASSIFIER

print(f"RF Accuracy: {accuracy_RF:.4f}")
print(f"RF AUC: {auc_RF:.4f}")


# # Predict the probability for the test file and submission  

# From my point of view, I will submit a logistic regression model with TF_IDF because it is simple and efficient for unbalanced large dataset also it offers high accuracy, and better handling of multiclass classification, in addition to its balanced performance across different types of toxicity which will improve the accuracy of prediction 

# In[62]:


#Iwill read the dataframe for the test file by Panda 

test_df = pd.read_excel(r'C:\Users\shoro\OneDrive\Desktop\submission_test.xlsx')


# In[63]:


# APPLY SAME STEPES FOR VIEW AND PROCESS DATA 

print(test_df.head())


# In[64]:


# CHECK THE NUMBER OF RECORDS AND COLUMNS
test_df.shape


# In[65]:


print (test_df.isnull().sum())


# In[66]:


# I WILL ADJUST THE NAME OF COLUMNS 
# I WILL Drop the first row (index 0) 
test_df = test_df.drop(0).reset_index(drop=True)


# In[67]:


print(test_df.head())


# In[68]:


# RENAME THE COLOMNS TO MATCH THE TRAINIG FILE 

test_df.columns = ['id', 'comment_text']

# RECHECK THE TEST_DF
print(test_df)


# In[69]:


#I will start the cleaning process 
#I Will first check the missing value in the dataset 

print (test_df.isnull().sum())


# In[72]:


# import the needed library
import re

# definea function to clean the text (same as what was disscused before.

def clean_text(text):
    # lowercase the text     
    text = text.lower()

    # remove URLs if present    
    text = re.sub(r'http\S+', '', text)

    # keep only letters      
    text = re.sub(r'[^a-zA-Z\s]', '', text)

   
    # treat spaces      
    text = re.sub(r'\s+', ' ', text).strip()

    return text 
   # finally apply the function to the comment_text 
test_df['comment_text'] = test_df['comment_text'].apply(clean_text)


# In[73]:


# I will check the process 
print(test_df[['comment_text']].head())


# In[74]:


# the changes will be saved to submission_test file 
test_df.to_excel(r'C:\Users\shoro\OneDrive\Desktop\submission_test.xlsx')


# In[89]:


test_df


# In[90]:


# i checked the missing value as in some steps may error occure
print (test_df.isnull().sum())


# In[ ]:





# In[91]:


# I will built a function to count number of token in the sentence similar to the previuos one 
# built for test_df 
# I need to call needed library( already called) 

import seaborn as sns
def total_tokens(sentence):
    
    #split by space  
    
    return len(sentence.split())

# Apply the function to each sentence in the 'comment_text'  

token_counts = test_df['comment_text'].apply(lambda x: total_tokens(str(x)))

# plot the result   

plt.figure(figsize=(10, 6))  #(10 inch width and 6 inch tall )

sns.histplot(token_counts, kde=True, bins=30) # I generated KDE curve 

plt.title("Token frequency per sentence")
plt.xlabel("Total Number of Tokens")
plt.ylabel("Frequency")
plt.show()


# In[94]:


# now I will apply the transform data to 'comment_text in' test_df
X_test_submission_tfidf = tfidf_vectorizer.transform(test_df['comment_text'])

# then I will predict the probability using logistic regression model
y_pred_prob_submission = logreg_model.predict_proba(X_test_submission_tfidf)

# finally we need to create new datafram called 'submissin' 
# and use it to prepare the predicted value for the submission file 

submission = pd.DataFrame({
    'id': test_df['id'],  # this unique identefier
    'comment_text': test_df['comment_text'],  # use the comment_text in the file 
    # now predict the probability for each type of toxicity 
    # the array has the shap (n-sample,n-classes) n started from zero:
    'toxic': y_pred_prob_submission[0][:, 1],  # this will drow the probability for 'toxic' type
    'severe_toxic': y_pred_prob_submission[1][:, 1],  # this will drow the probability for 'severe_toxic' type
    'obscene': y_pred_prob_submission[2][:, 1],  # this will drow the probability for 'obscene' type
    'threat': y_pred_prob_submission[3][:, 1],  # this will drow the probability for 'threat' type
    'insult': y_pred_prob_submission[4][:, 1],  # this will drow the probability for 'insult' type
    'identity_hate': y_pred_prob_submission[5][:, 1]   # this will drow the probability for 'identity_hate' type
})


# In[95]:


# I will check the first few rows to check if the process ok 
print(submission.head())


# In[96]:


# i will recheck for missing 
print (submission.isnull().sum())


# In[101]:


# now i will save the changes to the same file 
submission.to_excel(r'C:\Users\shoro\OneDrive\Desktop\submission_test.xlsx', index=False)


# In[ ]:





# In[102]:


# i will load the file to confirm the changes 
submission_cheking = pd.read_excel(r'C:\Users\shoro\OneDrive\Desktop\submission_test.xlsx')


# In[103]:


# recheck the missing 
print (submission_cheking.isnull().sum())


# In[104]:


# as missing values apears upon saving it could be to different reason 
# as long as 905 is less than 10% of the values in the dataset i will replace with 'unknown'
#
submission['comment_text'].fillna('unknown', inplace=True)



# In[105]:


# check 
print(submission['comment_text'].isnull().sum())


# In[106]:


submission.to_excel(r'C:\Users\shoro\OneDrive\Desktop\submission_test.xlsx', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




