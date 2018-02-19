from kivy.app import App
from kivy.properties import StringProperty, BooleanProperty, NumericProperty
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock, mainthread
from keras.callbacks import LambdaCallback, ModelCheckpoint
import json
import os
import betternn as nn
import configparser as cfg

class Modeller(Screen):
	corpus = StringProperty()
	seq_len = StringProperty()
	step = StringProperty()

	def onload(self):
		config = cfg.ConfigParser()
		config.sections()
		config.read('cache/input.ini')

		self.corpus = config['Modeller']['txt_corpus']
		self.seq_len = config['Modeller']['txt_seq_len']
		self.step = config['Modeller']['txt_step']

		print(self.corpus)

	def load_model(self):
		if not nn.Data.model == None:
			return
		try:
			nn.Data.vals = {
				'seq_len': int(self.ids['txt_seq_len'].text),
				'step': int(self.ids['txt_step'].text)
			}
		except ValueError:
			print("Could not parse a value!")
		nn.Data.model = nn.load_model(
			nn.Data.vals,
			self.ids['txt_corpus'].text
		)
		try:
			self.ids['txt_vals'].txt = json.dumps(nn.Data.vals, sort_keys=True, indent=4)
		except (ValueError, TypeError):
			print("Could not dump JSON")
			self.ids['txt_vals'].txt = str(nn.Data.vals)

		if not os.path.exists("cache"):
			os.makedirs("cache")
		if not os.path.exists("cache/input.ini"):
			f= open("cache/input.ini","w+")
			f.close()
		config = cfg.ConfigParser()
		config.sections()
		config.read('cache/input.ini')
		config['Modeller'] = {
			'txt_corpus': self.ids['txt_corpus'].text,
			'txt_seq_len': self.ids['txt_seq_len'].text,
			'txt_step': self.ids['txt_step'].text
		}
		with open("cache/input.ini", "w") as f:
			config.write(f)

class Trainer(Screen):
	_batchsize = StringProperty()
	_epochs = StringProperty()
	_weights = StringProperty()
	# prt = BooleanProperty()
	# fil = BooleanProperty()
	# wgt = BooleanProperty()

	def onload(self):
		config = cfg.ConfigParser()
		config.sections()
		config.read('cache/input.ini')

		b = config['Trainer']['txt_batch_size']
		print(type(b))

		# self._batchsize = str('128'),
		# self._epochs = str('60'),
		# self._weights = config['Trainer']['txt_weights'],
		# self.prt = bool(config['Trainer']['is_prt']),
		# self.fil = bool(config['Trainer']['is_fil']),
		# self.wgt = bool(config['Trainer']['is_wgt'])

	def train(self):
		# Try to store values
		try:
			nn.Data.vals['batch_size'] = int(self.ids['txt_batch_size'].text)
			nn.Data.vals['epochs'] = int(self.ids['txt_epochs'].text)
		except ValueError:
			print("Could not parse a value!")

		# Save config
		if not os.path.exists("cache"):
			os.makedirs("cache")
		if not os.path.exists("cache/input.ini"):
			f= open("cache/input.ini","w+")
			f.close()
		config = cfg.ConfigParser()
		config.sections()
		config.read('cache/input.ini')
		config['Trainer'] = {
			'txt_batch_size': self.ids['txt_batch_size'].text,
			'txt_epochs': self.ids['txt_epochs'].text,
			'txt_weights': self.ids['txt_weights'].text,
			'is_prt': self.ids['is_prt'].active,
			'is_fil': self.ids['is_fil'].active,
			'is_wgt': self.ids['is_wgt'].active
		}
		with open("cache/input.ini", "w") as f:
			config.write(f)

		# Load callbacks
		callbacks = []
		if self.ids['is_prt'].active:
			callbacks += [LambdaCallback(on_epoch_end=nn.prt_on_epoch_end)]
		if self.ids['is_fil'].active:
			callbacks += [LambdaCallback(on_epoch_end=nn.gen_on_epoch_end)]
		if self.ids['is_wgt'].active:
			out = self.ids['txt_weights'].text + "/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
			callbacks += [ModelCheckpoint(out, monitor='loss', verbose=1, save_best_only=True, mode='min')]

		# Train
		nn.train(nn.Data.model, nn.Data.vals, callbacks)

class Generator(Screen):
	generated = StringProperty()
	text = ""
	instance = None

	def init(self, **kwargs):
		super(kwargs)
		self.text = ""
		Generator.instance = self

	@mainthread
	def perCharCallback(self, c):
		self.ids["txt_output"].insert_text(c)

	def generate(self):
		metadata = str(self.ids["txt_metadata"].text)
		weights = str(self.ids["txt_weights"].text)
		self.ids["txt_output"].select_all()
		self.ids["txt_output"].delete_selection()
		self.ids["txt_output"].cursor = (0,0)

		# thread = Thread(target = nn.generateModelText, args=(metadata, weights, self.perCharCallback,))
		thread.run()

class MainMenu(Screen):
	def press(self, num):
		print('Number: {}'.format(num))

class MainApp(App):
	def build(self):
		pass

def ca(b):
	pass

if __name__ == '__main__':
	MainApp().run()
	# nn.trainNetwork("bnw/bnw.data.json", "bnw2", "bnw2/weights-improvement-18-1.7800.hdf5")
	# nn.generateModelText("bnw/bnw.data.json", "bnw2/weights-improvement-18-1.7800.hdf5")
