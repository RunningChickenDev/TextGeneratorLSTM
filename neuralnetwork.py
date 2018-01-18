import numpy
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

def createModel(X ,y):
	model = Sequential()
	model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='softmax'))
	return model

def drawModel(filepath):
	plot_model(createModel(), to_file=filepath)

def trainModel(infile, outfolder):
	# process text
	raw_text = open(infile).read()
	raw_text = raw_text.lower()
	# make text accessible to computer (mappings char - int)
	chars = sorted(list(set(raw_text)))
	char_to_int = dict((c, i) for i, c in enumerate(chars))
	int_to_char = dict((i, c) for i, c in enumerate(chars))
	# print some useful data
	n_chars = len(raw_text)
	n_vocab = len(chars)
	print("Total Characters: ", n_chars)
	print("Total Vocab: ", n_vocab)
	# print("Chars: ", chars)
	print("Char to Int: ", char_to_int)
	print("Int to Char: ", int_to_char)
	# prepare the actual dataset
	seq_length = 100	# prediction block size
	dataX = []			# prediction blocks
	dataY = []			# results (of predictions)
	for i in range(0, n_chars - seq_length, 1):
		seq_in = raw_text[i:i + seq_length]
		seq_out = raw_text[i + seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])
	n_patterns = len(dataX)
	# print("DataX: ", dataX)
	# print("DataY: ", dataY)
	print("Total Patterns: ", n_patterns)
	# reshape X to be [samples, time steps, features]
	X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
	# normalize
	X = X / float(n_vocab)
	# one hot encode the output variable
	y = np_utils.to_categorical(dataY)
	# define the LSTM model
	model = createModel(X, y)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# print("X: ", X)
	# print("y: ", y)
	# define the checkpoint, stops each iteration
	filepath=outfolder+"/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"	# name formatting
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	# GO
	plot_model(model, to_file="img/fit_model.png")
	model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

def generateText(infile, inweights, callback):
	# process text
	raw_text = open(infile).read()
	raw_text = raw_text.lower()
	# make text accessible to computer (mappings char - int)
	chars = sorted(list(set(raw_text)))
	char_to_int = dict((c, i) for i, c in enumerate(chars))
	int_to_char = dict((i, c) for i, c in enumerate(chars))
	# print some useful data
	n_chars = len(raw_text)
	n_vocab = len(chars)
	print("Total Characters: ", n_chars)
	print("Total Vocab: ", n_vocab)
	print("Chars: ", chars)
	print("Char to Int: ", char_to_int)
	print("Int to Char: ", int_to_char)
	# prepare the actual dataset
	seq_length = 100	# prediction block size
	dataX = []			# prediction blocks
	dataY = []			# results (of predictions)
	for i in range(0, n_chars - seq_length, 1):
		seq_in = raw_text[i:i + seq_length]
		seq_out = raw_text[i + seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])
	n_patterns = len(dataX)
	print("DataX: ", dataX)
	print("DataY: ", dataY)
	print("Total Patterns: ", n_patterns)
	# reshape X to be [samples, time steps, features]
	X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
	# normalize
	X = X / float(n_vocab)
	# one hot encode the output variable
	y = np_utils.to_categorical(dataY)
	# define the LSTM model
	model = createModel(X, y)
	print("X: ", X)
	print("y: ", y)
	# load the network weights
	model.load_weights(inweights)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	plot_model(model, to_file="img/gen_model.png")
	# pick a random seed
	start = numpy.random.randint(0, len(dataX)-1)
	pattern = dataX[start]
	print ("Seed:")
	print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
	# generate characters
	for i in range(1000):
		x = numpy.reshape(pattern, (1, len(pattern), 1))
		x = x / float(n_vocab)
		prediction = model.predict(x, verbose=0)
		index = numpy.argmax(prediction)
		result = int_to_char[index]
		seq_in = [int_to_char[value] for value in pattern]
		sys.stdout.write(result)
		callback(result)
		pattern.append(index)
		pattern = pattern[1:len(pattern)]
	print( "\nDone.")
