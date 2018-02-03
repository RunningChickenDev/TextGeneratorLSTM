from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import datetime
import numpy as np
import random
import sys
import io

# VALUES:
# - path
# - chars
# - char to int
# - maxlen (sequence size)
# - step
# - input/training sentences
# - output/training chars
# - model type
# -

path = "gedichten_mei/data.txt"
text = io.open(path, encoding='utf-8').read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
	sentences.append(text[i: i + maxlen])
	next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
	for t, char in enumerate(sentence):
		x[i, t, char_indices[char]] = 1
	y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
	# helper function to sample an index from a probability array
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)


def prt_on_epoch_end(epoch, logs):
	# Function invoked at end of each epoch. Prints generated text.
	print()
	print('----- Generating text after Epoch: %d' % epoch)

	start_index = random.randint(0, len(text) - maxlen - 1)
	for diversity in [0.2, 0.5, 1.0, 1.2]:
		print('----- diversity:', diversity)

		generated = ''
		sentence = text[start_index: start_index + maxlen]
		generated += sentence
		print('----- Generating with seed: "' + sentence + '"')
		sys.stdout.write(generated)

		for i in range(400):
			x_pred = np.zeros((1, maxlen, len(chars)))
			for t, char in enumerate(sentence):
				x_pred[0, t, char_indices[char]] = 1.

			preds = model.predict(x_pred, verbose=0)[0]
			next_index = sample(preds, diversity)
			next_char = indices_char[next_index]

			generated += next_char
			sentence = sentence[1:] + next_char

			sys.stdout.write(next_char)
			sys.stdout.flush()
		print()

generations_file = "gedichten_mei/gedichten.txt"

def gen_on_epoch_end(epoch, logs):
	# Function invoked at end of each epoch. Saves generated text in file.
	f = open(generations_file, "a")
	f.write("Epoch = {}:\n".format(epoch))
	start_index = random.randint(0, len(text) - maxlen - 1)
	for diversity in [0.2, 0.5, 1.0, 1.2]:
		f.write("  diversity = {}\n".format(diversity))
		generated = ''
		sentence = text[start_index: start_index + maxlen]
		generated += sentence
		f.write("  seed = {}\n".format(generated))
		f.write("  text:\n")
		f.write("-------------\n")

		f.write(generated)
		for i in range(400):
			x_pred = np.zeros((1, maxlen, len(chars)))
			for t, char in enumerate(sentence):
				x_pred[0, t, char_indices[char]] = 1.

			preds = model.predict(x_pred, verbose=0)[0]
			next_index = sample(preds, diversity)
			next_char = indices_char[next_index]

			generated += next_char
			sentence = sentence[1:] + next_char

			f.write(next_char)
			# sys.stdout.flush()
		f.write("\n-------------\n")
	f.write("\n-------------\n\n")
	f.close()

with open(generations_file, "a") as f:
	f.write('\n')
	f.write("Generated texts on {}\n".format(datetime.datetime.now()))
	f.write("==========\n")

out = "gedichten_mei/w/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# print_callback = LambdaCallback(on_epoch_end=prt_on_epoch_end)
write_callback = LambdaCallback(on_epoch_end=gen_on_epoch_end)
save_callback = ModelCheckpoint(out, monitor='loss', verbose=1, save_best_only=True, mode='min')

model.load_weights("gedichten_mei/w/weights-improvement-48-0.9858.hdf5")

if __name__ == "__main__":
	running = True
	while running:
		print("What to do? (train/generate)")
		choice = input("> ")
		if choice.lower() == "train":
			model.fit(x, y,
					  batch_size=128,
					  epochs=60,
					  callbacks=[write_callback, save_callback])
		elif choice.lower() == "generate":
			print("  Type seed:")
			seed = input(">>> ")[:40]
			print("  Using seed: '{}'".format(seed))
			print("  Type diversity:")
			diversity = float(input(">>> "))
			print("  Using diversity: {}".format(diversity))
			print("  --  ")
			generated = ''
			sentence = seed
			generated += sentence
			for i in range(400):
				x_pred = np.zeros((1, maxlen, len(chars)))
				for t, char in enumerate(sentence):
					x_pred[0, t, char_indices[char]] = 1.
				preds = model.predict(x_pred, verbose=0)[0]
				next_index = sample(preds, diversity)
				next_char = indices_char[next_index]
				generated += next_char
				sentence = sentence[1:] + next_char
				sys.stdout.write(next_char)
				sys.stdout.flush()
		elif choice.lower() == "exit":
			print("  Exitting...")
			running = False
		else:
			print("  Sorry, did not chatch that")
			print("  Try: train, generate, exit")
	print("Done")
