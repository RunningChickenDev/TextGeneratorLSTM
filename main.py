from kivy.app import App
from kivy.properties import StringProperty
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock, mainthread
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
	pass

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
