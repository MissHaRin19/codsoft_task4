import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

def predict_spam(message, vectorizer, model, X_train):
    data = pd.DataFrame({'text': [message]})

    data['text'] = data['text'].str.lower()

    ps = PorterStemmer()
    data['text'] = data['text'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))

    global stopwords
    stopwords = set(stopwords.words('english'))
    data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))

    X = vectorizer.transform(data['text']).toarray()
    y = np.zeros(1)

    y_pred = model.predict(X)

    return y_pred[0]

data = pd.read_csv('spam.csv', encoding='ISO-8859-1')



data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

data.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

encoder = LabelEncoder()
data['target'] = encoder.fit_transform(data['target'])



data = data.drop_duplicates(keep='first')

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(X_train).toarray()

model = XGBClassifier()
model.fit(X_train, y_train)

message = input("Enter a message: ")
prediction = predict_spam(message, vectorizer, model, X_train)

if prediction == 1:
    print("The message is not spam.")
else:
    print("The message is spam.")
    
