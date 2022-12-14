import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

train_df = pd.read_csv(r"C:\Users\Sercan\Desktop\MLProject\train.csv")
test_df =  pd.read_csv(r"C:\Users\Sercan\Desktop\MLProject\test.csv")

print(len(train_df))
train_df = train_df.drop_duplicates('text', keep='last')
print(len(train_df))

print(train_df['target'].value_counts())

EMOJIS = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}
URLPATTERN        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
USERPATTERN       = '@[^\s]+'
SEQPATTERN   = r"(.)\1\1+"
SEQREPLACE = r"\1\1"

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer as ps
from nltk.corpus import stopwords
import nltk
#nltk.download('wordnet')
wordLemm = WordNetLemmatizer()
ps = ps()

import re
def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)
    ### Replacing URL
    text = re.sub(URLPATTERN,' URL',text)
    ### Replacing EMOJI
    for emoji in EMOJIS.keys():
        text = text.replace(emoji, "EMOJI" + EMOJIS[emoji])  
    ### Replacing USER pattern
    text = re.sub(USERPATTERN,' URL',text)
    ### Removing non-alphabets
    text = re.sub('[^a-zA-z]'," ",text)
    ### Removing consecutive letters
    text = re.sub(SEQPATTERN,SEQREPLACE,text)
    text = text.lower()
    text = text.split()
    text = [wordLemm.lemmatize(word) for word in text if not word in stopwords.words('english') and len(word) > 1]
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

train_df['text'] = train_df['text'].apply(preprocess_text)

test_df ['text'] = test_df['text'].apply(preprocess_text)

# Bag of Words
cv = CountVectorizer(max_features=4000)
X = cv.fit_transform(train_df['text']).toarray()
y = train_df.iloc[:,4].values

# TF IDF Vectorizer
tfidf = TfidfVectorizer(max_features = 4000)
X1 = tfidf.fit_transform(train_df['text']).toarray()

from sklearn.model_selection import train_test_split


#CV split --- Train size = 0.7 Valid. size = 0.15 Test size = 0.15

X_train, x_temp, y_train, y_temp = train_test_split(X, y, stratify=y,
                                                       train_size=0.7,
                                                       random_state=3)

X_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, stratify=y_temp,
                                                  test_size=0.5,
                                                  random_state=3)


#TFIDF Vectorizer split --- Train size = 0.7 Valid. size = 0.15 Test size = 0.15

X1_train, x1_temp, y1_train, y1_temp = train_test_split(X1,   y,  stratify=y,
                                                       train_size=(0.7),
                                                       random_state=3)

X1_val, x1_test, y1_val, y1_test = train_test_split(x1_temp,
                                                  y1_temp,
                                                  stratify=y1_temp,
                                                  test_size=0.5,
                                                 random_state=3)


y1_test = np.delete(y1_test,1125,0)

#----------------GAUSSIAN NB--------------------#
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score
gnb = GaussianNB()

#   CV Naive Bayes
gnb.fit(X_train,y_train)

y_valpred = gnb.predict(X_val)

cm = confusion_matrix(y_val, y_valpred)
accuracy = accuracy_score(y_val , y_valpred)
print("Gaussian NB CM : ",cm)
print("Gaussian NB acc:", accuracy)



#  TF-IDF Gaussian NB
gnb.fit(X1_train, y1_train)

y1_predgnb = gnb.predict(X1_val)
cm = confusion_matrix(y1_test, y1_predgnb)
print(cm)
accuracy = accuracy_score(y1_test, y1_predgnb)

#------------------MULTINOMIAL NB-----------------
from sklearn.naive_bayes import MultinomialNB

mnb= MultinomialNB()
mnb.fit(X_train,y_train)

y_valpredmnb = mnb.predict(X_val)
cm2 = confusion_matrix(y_val, y_valpredmnb)
accuracy2 = accuracy_score(y_val , y_valpredmnb)
print("Multinomial NB CM : ", cm2)
print("Multinomial NB acc: ", accuracy2)


#TF-IDF MNB
mnb.fit(X1_train, y1_train)

y1_predmnb = mnb.predict(X1_val)
cm = confusion_matrix(y1_test, y1_predmnb)
print(cm)
accuracy = accuracy_score(y1_test, y1_predmnb)
print("Tfidf MNB Accuracy score: ", accuracy)



#-------------------KNN CLASSIFICATION----------------------
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
errorcv = []
errortf = []

for k in range(1,31):
    knn = KNeighborsClassifier(n_neighbors = k, metric='euclidean')
    knn.fit(X_train, y_train)
    y_valpredknn = knn.predict(X_val)
    
    cm3 = confusion_matrix(y_val, y_valpredknn)
    accuracy3 = accuracy_score(y_val , y_valpredknn)
    errorcv.append(mean_squared_error(y_val, y_valpredknn))
    
print("CV KNN CM for k in range 1-31 : ", cm3)
print("CV KNN accuracy for k in range 1-31 : ", accuracy3)
plt.plot(range(1,31),errorcv, label='For CV')
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


for k in range(1,31):
    knn = KNeighborsClassifier(n_neighbors = k, metric='euclidean')
    knn.fit(X1_train, y1_train)
    y1_valpredknntf = knn.predict(X1_val)
    
    cm3 = confusion_matrix(y1_val, y1_valpredknntf)
    accuracy3 = accuracy_score(y1_val , y1_valpredknntf)
    errortf.append(mean_squared_error(y1_val, y1_valpredknntf))
    
print("TFIDF KNN CM for k in range 1-31 : ", cm3)
print("TFIDF KNN accuracy for k in range 1-31 : ", accuracy3)
plt.plot(range(1,31),errortf , label = "for TFIDF")
plt.legend()


#----------------DECISION TREE--------------------------
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy', max_depth = 50, max_leaf_nodes=60 )

dtc.fit(X_train, y_train)

y_valpreddtc = dtc.predict(X_val)

cm5 = confusion_matrix(y_val, y_valpreddtc)
accuracy5 = accuracy_score(y_val , y_valpreddtc)
print("CV CM for DT : ", cm5)
print("CV accuracy for DT : ", accuracy5)

dtc.fit(X1_train, y1_train)

y1_valpreddtc = dtc.predict(X1_val)
cm8 = confusion_matrix(y1_val, y1_valpreddtc)
accuracy8 = accuracy_score(y1_val , y1_valpreddtc)
print("TFIDF CM for DT : ", cm8)
print("TFIDF accuracy for DT : ", accuracy8)


#------------------MULTILAYER PERCEPTRON------------------------
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(activation='relu', solver='sgd', hidden_layer_sizes=(5,10),learning_rate='adaptive',max_iter = 600,learning_rate_init=0.13, random_state=1)

nn.fit(X_train, y_train)

y_valpredbp = nn.predict(X_val)

cm4 = confusion_matrix(y_val, y_valpredbp)
accuracy4 = accuracy_score(y_val , y_valpredbp)
print("CM for BP : ", cm4)
print("accuracy for BP : ", accuracy4)


nn.fit(X1_train, y1_train)

y1_valpredbp = nn.predict(X1_val)
cm4 = confusion_matrix(y1_val, y1_valpredbp)
accuracy4 = accuracy_score(y1_val , y1_valpredbp)
print("TFIDF CM for BP : ", cm4)
print("TFIDF accuracy for BP : ", accuracy4)



#----------------------K FOLD CROSS VALIDATION---------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

skf5 = StratifiedKFold(n_splits = 5,random_state=None)

skf5.get_n_splits(X,y)

acc_gnb =[]
acc_mnb =[]
acc_knn =[]
acc_nn =[]
acc_dtc =[]
i = 1

cm1 = []
cm2 = []
cm3 = []
cm4 = []
cm5 = []

for train_index, test_index in skf5.split(X,y):
    print("iteration ", i)
    
    X2_train, X2_test = X[train_index], X[test_index]
    y2_train, y2_test = y[train_index], y[test_index]
    knn = KNeighborsClassifier(n_neighbors = 30, metric='euclidean')
    
    gnb.fit(X2_train, y2_train)
    mnb.fit(X2_train, y2_train)
    knn.fit(X2_train, y2_train)
    nn.fit(X2_train, y2_train)
    dtc.fit(X2_train, y2_train)
    
    
    predgnb = gnb.predict(X2_test)
    score_gnb = accuracy_score(predgnb,y2_test)
    cmgnb = confusion_matrix(y2_test, predgnb)
    cm1.append(cmgnb)
    acc_gnb.append(score_gnb)
    
    
    predmnb = mnb.predict(X2_test)
    score_mnb = accuracy_score(predmnb,y2_test)
    cmmnb = confusion_matrix(y2_test, predmnb)
    cm2.append(cmmnb)
    acc_mnb.append(score_mnb)
    
    
    predknn = knn.predict(X2_test)
    score_knn = accuracy_score(predknn,y2_test)
    cmknn = confusion_matrix(y2_test,predknn)
    cm3.append(cmknn)
    acc_knn.append(score_knn)
    
    
    prednn = nn.predict(X2_test)
    score_nn = accuracy_score(prednn,y2_test)
    print("NN loss for iteration ", i ," : ", nn.loss_)
    acc_nn.append(score_knn)
    cmnn = confusion_matrix(y2_test,prednn)
    cm4.append(cmknn)
    


    preddtc = dtc.predict(X2_test)
    score_dtc = accuracy_score(preddtc,y2_test)
    
    acc_dtc.append(score_dtc)
    cmdtc = confusion_matrix(y2_test,preddtc)
    cm5.append(cmdtc)
    i +=1


print("K=5 kFold gnb acc: " , np.array(acc_gnb).mean())
print("K=5 kFold gnb CM: ", sum(cm1))
print("K=5 kFold mnb acc: " ,np.array(acc_mnb).mean())
print("K=5 kFold mnb CM: ", sum(cm2))
print("K=5 kFold knn acc: " , np.array(acc_knn).mean())
print("K=5 kFold knn CM: ", sum(cm3))
print("K=5 kFold nn acc: " , np.array(acc_nn).mean())
print("K=5 kFold nn CM: ", sum(cm4))
print("K=5 kFold dtc acc: " , np.array(acc_dtc).mean())
print("K=5 kFold dtc CM: ", sum(cm5))