import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

df=pd.read_csv("data.csv")

japanese_stop_words = ['あの', 'いくつか', 'いつ', 'そして', 'その', 'たくさん', 'だから', 'とても', 'どうして', 'なぜ']
arabic_stop_words = ['أنا', 'هو', 'في', 'من', 'على', 'إلى', 'هذا', 'الذي', 'الذين', 'إذا']

sw=["au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en", "et", "eux", "il", "je", "la", "le", "leur", "lui", "ma", "mais", "me", "même", "mes", "moi", "mon", "ne", "nos", "notre", "nous", "on", "ou", "par", "pas", "pour", "qu", "que", "qui", "sa", "se", "ses", "son", "sur", "ta", "te", "tes", "toi", "ton", "tu", "un", "une", "vos", "votre", "vous"
]+list(ENGLISH_STOP_WORDS)+japanese_stop_words+arabic_stop_words

def preprocess_text(text):
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in sw])
    return text

df['Description'] = df['Description'].apply(preprocess_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Description'])
y_cost = df['Cost Center']
y_gl = df['GL Account']

clf_cost = svm.SVC(kernel='linear')
clf_cost.fit(X, y_cost)

clf_gl = svm.SVC(kernel='linear')
clf_gl.fit(X, y_gl)

def svmmodel(description):

    
    description = description.split(",")
    for desc in description:
        desc = preprocess_text(desc)
        desc = [desc]
        X_input = vectorizer.transform(desc)

        predicted_cost = clf_cost.predict(X_input)
        predicted_gl = clf_gl.predict(X_input)

        output = [predicted_cost,predicted_gl]

        
    return output
        



'''for i in df.index:
    val=df.loc[i,'Value']
    df.loc[i,"ylabel"]=str(val)

df['Description'] = df['Description'].str.lower()

x_train=df['Description'].tolist()

y_train=df['ylabel'].tolist()

tfdif=TfidfVectorizer(stop_words=sw)

train_vectors=tfdif.fit_transform(x_train)

svm = OneVsRestClassifier(SVC(kernel='sigmoid', probability=True))
svm.fit(train_vectors, y_train)


def svmmodel(description):
    description_vector=tfdif.transform([description])
    test_probabilities = svm.predict_proba(description_vector)
    anomaly_scores = [1 - max(probs) for probs in test_probabilities]
    for i in anomaly_scores:
        if i>0.77:
            output=svm.predict(description_vector)
            output=str(output)
            return(output)
    else:
            output=svm.predict(description_vector)
            output=str(output)
            output=output[2:len(output)-2]
            output=list(output.split(','))'''

