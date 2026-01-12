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
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

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

# NEXT WORD PREDICTOR USING LSTM

sample_text = """Data second-hand in these extents frequently have inferior 1% of excellent, but “entertaining” occurrences (for example fraudsters utilizing credit cards, consumer clicking poster or debased attendant thumbing through allure network). 
However, most machine intelligence algorithms do to malfunction very well accompanying unstable datasets. 
The following seven methods can help you, to train a classifier to discover the atypical class.
"""

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

X = padded_sequence[:,:-1]
y = padded_sequence[:,-1]
print(tokenizer.word_index)

y = to_categorical(y, num_classes=len(tokenizer.word_index)+1)
print(X.shape, y.shape)

model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, 10, input_length=max_len-1))
model.add(LSTM(100))
model.add(Dense(len(tokenizer.word_index)+1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.fit(X, y, epochs=100, verbose=1))

def predict_next_word(model, tokenizer, text, max_sequence_len):
    tokenized_text = tokenizer.texts_to_sequences([text])[0]
    tokenized_text = pad_sequences([tokenized_text], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(tokenized_text, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return ""

seed_text = "most machine intelligence"
next_word = predict_next_word(model, tokenizer, seed_text, max_len)
print(f"Next word prediction for '{seed_text}: {next_word}'")



