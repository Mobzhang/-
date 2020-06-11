import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
def prece_data():   #对情感文本数据进行预处理，返回text以及label
    pos_text = pd.read_table(r'data\positif.txt',names=['text'],encoding='utf-8')
    neg_text = pd.read_table(r'data\negatif.txt',names=['text'],encoding='utf-8')
    pos_text['label'] = 1
    neg_text['label'] = 0
    pos_text['cuted'] = pos_text['text'].map(lambda x: ' '.join(jieba.cut(x)))
    neg_text['cuted'] = neg_text['text'].map(lambda x: ' '.join(jieba.cut(x)))
    frames = [pos_text, neg_text]
    all_text = pd.concat(frames)
    x = all_text['cuted']
    y = all_text['label']
    return x, y
def stopwords_list():
    with open(r'stopwords-master\cn_stopwords.txt',encoding='utf-8') as f:
        lines = f.readlines()
        result = [i.strip('\n') for i in lines]
    return result
classifiers = {'KNN': KNeighborsClassifier(),
                'LR': LogisticRegression(),
                'RF': RandomForestClassifier(),
                }
method = ['KNN', 'LR', 'RF']
stopwords = stopwords_list()
X, y = prece_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=18)  #对数据集进行划分
print(X_train.shape)
vect = CountVectorizer()
vect.fit(X_train)
print(len(vect.vocabulary_))
vect1 = CountVectorizer(max_df=0.8, min_df=5, stop_words=stopwords,
                        token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b')
vect1.fit(X_train)
print(len(vect1.vocabulary_))
y = []
for i in method:
    result = []
    print("method is :",i)
    clf = classifiers.get(i)
    clf.fit(vect.transform(X_train), y_train)
    y_pred = clf.predict(vect.transform(X_test))
    # print('default result')
    # print(f1_score(y_test, y_pred))
    result.append(f1_score(y_test, y_pred))
    clf.fit(vect1.transform(X_train), y_train)
    y_pred = clf.predict(vect1.transform(X_test))
    # print('stop word result')
    # print(f1_score(y_test, y_pred))
    result.append(f1_score(y_test, y_pred))
    pipe = make_pipeline(TfidfVectorizer(min_df=5), clf)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    # print('Tfidf result')
    # print(f1_score(y_test, y_pred))
    result.append(f1_score(y_test, y_pred))
    y.append(result)
x = ['default', 'stop_word', 'Tfidf']
style = ['-o', '--o', '-.o']
for i in range(0,3):
    plt.plot(x,y[i],style[i],label=method[i])
plt.ylim(0.65, 1)
plt.xlabel('Classfication method')
plt.ylabel('F1-score')
plt.legend()
plt.savefig('F1-score.png')
plt.show()
