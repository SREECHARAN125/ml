import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from fuzzywuzzy import process



df = pd.read_csv("pro1/data.csv")

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = ''.join(char for char in text if not char.isdigit())
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['Description'] = df['Description'].apply(preprocess_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Description'])
y_cost = df['Cost Center']
y_gl = df['GL Account']

clf_cost = svm.SVC(kernel='sigmoid')
clf_cost.fit(X, y_cost)

clf_gl = svm.SVC(kernel='sigmoid')
clf_gl.fit(X, y_gl)

def svm(description):
  #description = description.lower()
  #description = description.split(",")
  data = []
  for text in description:
    text = text.lower()
    closest_match = process.extractOne(text, df['Description'])
    if closest_match[1] < 65:
      return (-1,-1)
    else:
      X_input = vectorizer.transform([closest_match[0]])
      predicted_cost = clf_cost.predict(X_input)
      predicted_gl = clf_gl.predict(X_input)
      #closest_match[0] = str(closest_match[0])
      #predicted_cost = str(predicted_cost)
      #predicted_gl = str(predicted_gl)
      data.append([predicted_cost,predicted_gl,closest_match[0]])  
  return data
  
