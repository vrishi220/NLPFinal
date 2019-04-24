import csv, pickle, pandas
import matplotlib.pyplot as plt
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

batch_size, dropout, emb_dim, epochs, num_words= 50, 0.7, 128, 10, 8000

fnames = ["#anxiety","#depression","#suicide","#neutral"]
df = pandas.read_csv("u2_info.csv",names=['CATEGORY','TITLE'],encoding="latin1")

tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['TITLE'].values)
sequences = tokenizer.texts_to_sequences(df['TITLE'].values)
word_index = tokenizer.word_index

df['LABEL'] = 0

for v,each in enumerate(fnames): df.loc[df['CATEGORY'] == each, 'LABEL'] = v

X = pad_sequences(sequences, maxlen=130)

X_train, X_test, y_train, y_test = train_test_split(X, to_categorical(df['LABEL'], num_classes=4), test_size=0.25, random_state=42)

model = Sequential()
model.add(Embedding(num_words, emb_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(dropout))
model.add(LSTM(64, dropout=dropout, recurrent_dropout=dropout))
model.add(Dense(len(fnames), activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())
h = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',patience=7, min_delta=0.0001)])

accr = model.evaluate(X_test,y_test)

plotaccuracy = plt.plot(range(1,epochs+1),h.history['acc'],range(1,epochs+1),h.history['val_acc'])
plt.xlabel('Epochs'); plt.ylabel('Accuracy')
plt.legend(('Train Accuracy','Test Accuracy'))
plt.show(plotaccuracy)

print('List of Historical val_loss Scores: {}\nList of Historical val_acc Scores: {}\nTest Loss: {}\nTest Accuracy: {}'.format(h.history['val_loss'],h.history['val_acc'],accr[0],accr[1]))
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

pickle.dump(model, open('NLP6.sav', 'wb'))

