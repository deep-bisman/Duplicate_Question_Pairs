import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv('questions.csv')
#print(df.head(10))
#print(df.shape)
#print(df.info())
#print(df.isnull().sum())

# distribution of duplicate and non-duplicate questions

#print(df['is_duplicate'].value_counts().sum())
x=((df['is_duplicate'].value_counts(normalize=True))*100).plot(kind='bar')
#plt.show()



# Repeated Questions
qid=pd.Series(df['qid1'].tolist()+df['qid2'].tolist())
#print(qid.shape[0])
#print('total no of unique questions are :',np.unique(qid).shape[0])
n=qid.value_counts()>1
#print('total no of unique questions that are repeated :',n[n].shape[0])
#print(np.unique(qid).shape[0]+n[n].shape[0])

# this syntax is used to find the total no. of repeated questions:
#print((qid.value_counts() - 1).sum())





# now we gonna try simple technique as applying bag of words in cloumn question1 and question2 and
#then we will find the cosine similarity between them and then we will apply some threshold to find out whether they are duplicate or not.

# to make computation fast we will take only 10000 rows of data

'''new_df=df.sample(30000)
(new_df.shape)
(new_df.head(10))
(new_df.isnull().sum())
(new_df.duplicated().sum())


ques_df=new_df[['question1','question2']]
#print(ques_df.isnull().sum())
#print(ques_df.duplicated().sum())
#print(ques_df.isna().sum())
from sklearn.feature_extraction.text import CountVectorizer
# merge texts
questions = list(ques_df['question1']) + list(ques_df['question2'])

cv = CountVectorizer(max_features=3000)
q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(),2)
temp_df1 = pd.DataFrame(q1_arr, index= ques_df.index)
temp_df2 = pd.DataFrame(q2_arr, index= ques_df.index)
temp_df = pd.concat([temp_df1, temp_df2], axis=1)
#print(temp_df.shape)

temp_df['is_duplicate'] = new_df['is_duplicate']
#print(temp_df.head(10))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(temp_df.iloc[:,0:-1].values,temp_df.iloc[:,-1].values,test_size=0.2,random_state=1)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
print(accuracy_score(y_test,y_pred))'''





# using some feature engineering techniques to improve the accuracy of our model


new_df = df.sample(30000,random_state=2)

# Feature Engineering

#calculate the length of quest.1 and quest.2 each letter in each word
new_df['q1_len'] = new_df['question1'].str.len() 
new_df['q2_len'] = new_df['question2'].str.len()
#print(new_df.head())


#calculate the number of words in each question
new_df['q1_num_words'] = new_df['question1'].apply(lambda row: len(row.split(" ")))
new_df['q2_num_words'] = new_df['question2'].apply(lambda row: len(row.split(" ")))
new_df.head()



def common_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return len(w1 & w2)
new_df['word_common'] = new_df.apply(common_words, axis=1)
new_df.head()


def total_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return (len(w1) + len(w2))
new_df['word_total'] = new_df.apply(total_words, axis=1)
new_df.head()

new_df['word_share'] = round(new_df['word_common']/new_df['word_total'],2)
#print(new_df.head())

final_df = new_df.drop(columns=['id','qid1','qid2','question1','question2'])
#print(final_df.shape)
#print(final_df.head())


from sklearn.feature_extraction.text import CountVectorizer
ques_df=new_df[['question1','question2']]
# merge texts
questions = list(ques_df['question1']) + list(ques_df['question2'])

cv = CountVectorizer(max_features=3000)
q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(),2)
temp_df1 = pd.DataFrame(q1_arr, index= ques_df.index)
temp_df2 = pd.DataFrame(q2_arr, index= ques_df.index)
temp_df = pd.concat([temp_df1, temp_df2], axis=1)
temp_df.shape

final_df = pd.concat([final_df, temp_df], axis=1)
#print(final_df.shape)
final_df.head()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(final_df.iloc[:,1:].values,final_df.iloc[:,0].values,test_size=0.2,random_state=1)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
#print(accuracy_score(y_test,y_pred))








