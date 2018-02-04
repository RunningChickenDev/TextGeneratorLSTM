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
import string

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

# CONSTANTS:
class Data:
	vals_keys_list = ('text', 'chars', 'char_indx', 'indx_char', 'seq_len', 'step', 'train_x', 'train_y')
	model = None
	vals = None
	gens_file = "gedichten_mei/gedichten.txt"
	gens_header = True

class Writer:
	def __init__(self, enabled=True, out=None):
		self.enabled = enabled
		if out == None:
			self.out = sys.stdout
		else:
			self.out = out

	def msg(self, msg, master=""):
		if self.enabled:
			self.out.write("[{:<12}] {}\n".format(master[:12], msg))

	def msg_val(self, name, val, master=""):
		if self.enabled:
			self.out.write("[{:<12}] {}: {}\n".format(master[:12], name, val))

w = Writer()

def analyze_data(path, vals, extended=False):
	if vals == None:
		vals = {}

	# Read the entire text
	text = ""
	read = 0
    # text = io.open(path, encoding='utf-8').read().lower()
	try:
		text = io.open(path, encoding='ascii').read().lower()
		# with open(path, "r") as f:
		# 	for line in f:
		# 		for c in line:
		# 			if ord(c) < 128:
		# 				text += c
		# 				read += 1
		# 			elif True:
		# 				w.msg_val("Could not read", c, "Analyze")
		# 				print("Read: ", read)
	except FileNotFoundError:
		w.msg_val("Could not read path: ", path, "Analyze")
		return


	w.msg_val('corpus length:', str(len(text)), "Analyze")
	vals['text'] = text

	# create the dicts
	w.msg('Registering characters ...', "Analyze")
	chars = sorted(list(set(text)))
	w.msg_val('Total chars', str(len(chars)), "Analyze")
	char_indices = dict((c, i) for i, c in enumerate(chars))
	indices_char = dict((i, c) for i, c in enumerate(chars))
	if extended:
		w.msg_val('Char to indices:', str(char_indices), "Analyze")
	vals['chars'] = chars
	vals['char_indx'] = char_indices
	vals['indx_char'] = indices_char

	# cut the text in semi-redundant sequences of maxlen characters
	w.msg('Analyzing training data ...', "Analyze")
	seq_len = vals['seq_len'] if 'seq_len' in vals else 40	# PyMagic
	step = vals['step'] if 'step' in vals else 3
	sentences = []
	next_chars = []
	for i in range(0, len(text) - seq_len, step):
		sentences.append(text[i: i + seq_len])
		next_chars.append(text[i + seq_len])
	w.msg_val('Total train sequences:', len(sentences), "Analyze")
	vals['seq_len'] = seq_len
	vals['step'] = step

	w.msg('Vectorization ...', "Analyze")
	x = np.zeros((len(sentences), seq_len, len(chars)), dtype=np.bool)
	y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
	for i, sentence in enumerate(sentences):
		for t, char in enumerate(sentence):
			x[i, t, char_indices[char]] = 1
		y[i, char_indices[next_chars[i]]] = 1
	vals['train_x'] = x	# FIXME might cause issues with dumping
	vals['train_y'] = y

	w.msg('Done', "Analyze")
	return vals

def is_vals_complete(vals):
	if all (k in vals for k in Data.vals_keys_list):
		return True
	else:
		return False

def load_model(vals=None, path=None, weights=None):
	if vals == None:
		if path == None:
			w.msg("Cannot load model: no path to data and no previous data!", "Load")
			return
		w.msg("No values found, analyzing ...", "Load")
		vals = analyze_data(path, vals)
	if not is_vals_complete(vals):
		w.msg("Values seem to be missing, re-creating ...", "Load")
		vals = analyze_data(vals, path, repair=True)

	w.msg("Building model ...", "Load")
	model = Sequential()
	model.add(LSTM(128, input_shape=(vals['seq_len'], len(vals['chars']))))
	model.add(Dense(len(vals['chars'])))
	model.add(Activation('softmax'))
	optimizer = RMSprop(lr=0.01)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)

	if not weights == None:
		w.msg("Loading weights", "Load")
		model.load_weights("gedichten_mei/w/weights-improvement-48-0.9858.hdf5")

	w.msg("Done", "Load")
	Data.vals = vals
	return model

def train(model, vals, callbacks=[]):
	if model == None:
		w.msg("No model!", "Train")
		return
	if vals == None:
		w.msg("No values!", "Train")
		return
	if not 'train_x' in vals:
		w.msg("Value 'train_x' missing!", "Train")
		return
	if not 'train_y' in vals:
		w.msg("Value 'train_y' missing!", "Train")
		return
	model.fit(
		x, y,
		batch_size = vals['batch_size'] if 'batch_size' in vals else 128,
		epochs = vals['epochs'] if 'epochs' in vals else 60,
		callbacks = callbacks
	)

def generate(model, vals, out, seed=None, epoch=None, diversities=[0.2, 0.5, 1.0, 1.2]):
	out.write("Epoch = {}:\n".format(epoch))
	start_index = random.randint(0, len(vals['text']) - vals['seq_len'] - 1)
	for diversity in diversities:
		out.write("  diversity = {}\n".format(diversity))
		generated = ''
		sentence = vals['text'][start_index: start_index + vals['seq_len']] if seed == None else str(seed)[:vals['seq_len']]
		generated += sentence
		out.write("  seed = '{}'\n".format(generated))
		out.write("  text:\n")
		out.write("-------------\n")

		out.write(generated)
		out.flush()
		for i in range(400):
			x_pred = np.zeros((1, vals['seq_len'], len(vals['chars'])))
			for t, char in enumerate(sentence):
				x_pred[0, t, vals['char_indx'][char]] = 1.

			preds = model.predict(x_pred, verbose=0)[0]
			next_index = sample(preds, diversity)
			next_char = vals['indx_char'][next_index]

			generated += next_char
			sentence = sentence[1:] + next_char

			out.write(next_char)
			out.flush()
		out.write("\n-------------\n")	# after generating one diversity
		out.flush()
	out.write("\n-------------\n\n")	# after generating (one epoch)
	out.flush()

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
	# print()
	# print('----- Generating text after Epoch: %d' % epoch)
    #
	# start_index = random.randint(0, len(text) - maxlen - 1)
	# for diversity in [0.2, 0.5, 1.0, 1.2]:
	# 	print('----- diversity:', diversity)
    #
	# 	generated = ''
	# 	sentence = text[start_index: start_index + maxlen]
	# 	generated += sentence
	# 	print('----- Generating with seed: "' + sentence + '"')
	# 	sys.stdout.write(generated)
    #
	# 	for i in range(400):
	# 		x_pred = np.zeros((1, maxlen, len(chars)))
	# 		for t, char in enumerate(sentence):
	# 			x_pred[0, t, char_indices[char]] = 1.
    #
	# 		preds = model.predict(x_pred, verbose=0)[0]
	# 		next_index = sample(preds, diversity)
	# 		next_char = indices_char[next_index]
    #
	# 		generated += next_char
	# 		sentence = sentence[1:] + next_char
    #
	# 		sys.stdout.write(next_char)
	# 		sys.stdout.flush()
	# 	print()
	generate(Data.model, Data.vals, sys.stdout, epoch=epoch)

def gen_on_epoch_end(epoch, logs):
	# Function invoked at end of each epoch. Saves generated text in file.
	f = open(Data.gens_file, "a")
	if Data.gens_header:
		f.write('\n')
		f.write("Generated texts on {}\n".format(datetime.datetime.now()))
		f.write("==========\n")
		Data.gens_header = False
	generate(Data.model, Data.vals, f, epoch=epoch)
	f.close()

# TODO: Move these to caller (with private confs)
# out = "gedichten_mei/w/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# # print_callback = LambdaCallback(on_epoch_end=prt_on_epoch_end)
# write_callback = LambdaCallback(on_epoch_end=gen_on_epoch_end)
# save_callback = ModelCheckpoint(out, monitor='loss', verbose=1, save_best_only=True, mode='min')

# This is in the case of cmd
if __name__ == "__main__":
	print("Type path to data:")
	path = input("> ")
	print("Type weights to load (leave empty for none)")
	weights = input("> ")
	if weights == "":
		weights = None
	Data.model = load_model(Data.vals, path, weights)

	running = True
	while running:
		print("What to do? (train/generate)")
		choice = input("> ")
		if choice.lower() == "train":
			print("  Type weights output:")
			out = input(">>> ")
			print("  Using weights: '{}'".format(out))
			print("  (Data will be output on console)")
			print("  --  ")
			train(Data.model, Data.vals, sys.stdout, callbacks=[
				LambdaCallback(on_epoch_end=prt_on_epoch_end),
				ModelCheckpoint(out, monitor='loss', verbose=1, save_best_only=True, mode='min')
			])
			print("  --  ")
		elif choice.lower() == "generate":
			print("  Type seed:")
			raw_seed = input(">>> ")
			seed = ""
			for c in raw_seed:
				if c in Data.vals['char_indx'].keys():
					seed += c
			print("  Using seed: '{}'".format(seed))
			print("  Type diversity:")
			diversity = float(input(">>> "))
			print("  Using diversity: {}".format(diversity))
			print("  --  ")
			generate(Data.model, Data.vals, sys.stdout, seed=seed, epoch=-1, diversities=[diversity])
		elif choice.lower() == "exit":
			print("  Exitting...")
			running = False
		else:
			print("  Sorry, did not chatch that")
			print("  Try: train, generate, exit")
	print("Done")
