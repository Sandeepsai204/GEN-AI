import pandas as pd
import numpy as np
import tensorflow as tf
import spacy
import gensim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from spacy import displacy
from spacy.tokens import Span
# from gensim.models import Word2Vec, KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Bag of Words and Training the Model
dataFrame = pd.read_csv("Data/spam.csv")
dataFrame["Spam"] = dataFrame['Category'].apply(lambda x: 1 if x == "spam" else 0)

X_train, X_test, Y_train, Y_test = train_test_split(dataFrame.Message, dataFrame.Spam, test_size=0.2)

vector = CountVectorizer()

X_train_cv = vector.fit_transform(X_train.values)
X_test_cv = vector.transform(X_test)

X_train_np = X_train_cv.toarray()


model = MultinomialNB()
model.fit(X_train_cv, Y_train)

Y_pred = model.predict(X_test_cv)
message = {"Loan application of Rs. 1,00,000 is shortlisted confirm now"}

message_cnt = vector.transform(message)
# print(model.predict(message_cnt))

corpus = [
    "Thor Eating Pizza, Loki is Eatin Pizza, Ironman ate Pizza already",
    "Apple is Anouncing new iphone tomorrow",
    "Tesla is Anouncing new model-5 tomorrow",
    "Google is Anouncing new pixel tomorrow",
    "Amazon is Anouncing new eco dot tomorrow",
    "Microsoft is Anouncing new surface tomorrow",
    "I am eating Biryani and you are eating Grapes",
]

# Term Frequency and Inverse Document Frequency (TF-IDF)
dataFrame = pd.read_csv("Data/ecom_data.csv")

dataFrame['label_num'] = dataFrame['label'].map({
    'Household': 0,
    'Electronics': 1,
    'Cloths & Accessories': 2,
    'Books': 3
})

# print(dataFrame.head(10))
X_train, X_test, Y_train, Y_test = train_test_split(dataFrame.product_name, dataFrame.label_num, test_size=0.2)

tf = TfidfVectorizer()

X_train_tf = tf.fit_transform(X_train)
X_test_tf = tf.transform(X_test)


clf = DecisionTreeClassifier()
clf.fit(X_train_tf, Y_train)
new_y_pred = clf.predict(X_test_tf)

message = ["pyjama"]
message_cnt = tf.transform(message)
# print(clf.predict(message_cnt))

nlp = spacy.load('en_core_web_sm')
pipes = nlp.pipe_names
text = nlp("Tesla inc is going to acquire twitter at forty five billion dollars")

s1 = Span(text, 0,2, label="ORG")
s2 = Span(text, 8,12, label="ORG")

text.set_ents([s1, s2], default='unmodified')

# for ent in text.ents:
    # print(ent.text, ' | ', ent.label_, ' | ', spacy.explain(ent.label_))
# displacy.render(text, style="ent")

# NEXT WORD PREDICTOR USING LSTM

sample_text = """Data is often called the new oil of the digital age, and for a good reason.
It fuels decision-making processes, drives innovation, and helps organizations gain insights into their operations.
However, before data can be harnessed for these purposes, it must go through a rigorous data cleaning process.
In this article, we'll explore the essential steps and strategies involved in data cleaning from the perspective of a data analyst."""

# Initiate the tokenizer
tokenizer = Tokenizer()

tokenizer.fit_on_texts([sample_text])

input_sequences = []

for sentence in sample_text.split("\n"):
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
    
    for index in range(1, len(tokenized_sentence)):
        input_sequences.append(tokenized_sentence[:index+1])

max_len = max([len(x) for x in input_sequences])
padded_sequence = pad_sequences(input_sequences, maxlen = max_len, padding='pre')
print(padded_sequence)



