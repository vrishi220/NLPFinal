import pickle, numpy, sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
classes = ["Anxiety","Depression","Suicide","Neutral"]

def gen(e):
	initia = numpy.array([0.,0.,0.,0.])
	for s in e:
		s = numpy.array([s])
		tokenizer = Tokenizer(num_words=8000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
		tokenizer.fit_on_texts(s)
		a = pad_sequences(tokenizer.texts_to_sequences(s), maxlen=130)
		at = pickle.load(open('NLP5.sav', 'rb'))
		res = at.predict(a)
		initia += at.predict(a)[0]
	initia /= len(e)
	print("\n=====\n")
	for v,i in enumerate(initia): print("\t{} : {}\n".format(classes[v],round(i*100,2)))
	print("\n=====\n")

	

d = ""
sentences = input("\n\nTell us how you are feeling :) \n\n A:\t")
Q = sentences.split('.')
if '' in Q: Q.remove('')
gen(Q)
