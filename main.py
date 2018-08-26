from kivy.app import App
from kivy.properties import StringProperty, BooleanProperty, NumericProperty
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.config import ConfigParser
from kivy.uix.settings import Settings
from keras.callbacks import LambdaCallback, ModelCheckpoint
from io import StringIO
import betternn as nn
import configparser as cfg
import sys
from threading import Thread
import time

config = ConfigParser()
config.read('gui/config.ini')

#TODO: put this somewhere else
class FlushableStringIO(StringIO):
	def __init__(self, flush_call, initial_value='', newline='\n'):
		super().__init__(initial_value, newline)
		self._flush_call_ = flush_call

	def flush(self):
		self._flush_call_(self)

class LoadModelScreen(Screen):
	def do_it(self):
		def flush_call(fsio):
			self.ids['feedback'].text = fsio.getvalue()
			self.ids['feedback'].texture_update()
		out = FlushableStringIO(flush_call=flush_call)

		nn.w.out = out

		in_data = config['Model']['in_data']
		in_weights = config['Model']['in_weights']
		if in_weights[-5:] != ".hdf5":
			in_weights = None
		model_type = config['Model']['model_type']

		nn.Data.model = nn.load_model(nn.Data.vals, in_data, in_weights, model_type)
		# t = Thread(target = nn.load_model, args = (nn.Data.vals, in_data, in_weights, ))
		# t.start()
		# t.join()
		out.close()


		nn.w.out = sys.stdout

		# nn.Data.vals['']
		# "[color=#4488ff][INF] Loading Model ...[/color]\n"

class TrainScreen(Screen):
	def do_it(self):
		callbacks = []

		if config["Training"]["callback_print"] == "1":
			callbacks += [LambdaCallback(on_epoch_end=nn.prt_on_epoch_end)]
			print("Added print callback")
		if config["Training"]["callback_tofile"] == "1":
			nn.Data.gens_file = str(config["Training"]["callback_tofile_file"])
			callbacks += [LambdaCallback(on_epoch_end=nn.gen_on_epoch_end)]
			print("Added file callback")
		if config["Training"]["callback_weights"] == "1":
			out = ""
			out += config["Training"]["callback_weights_folder"]
			if out[-1:] != '/' or out[-1:] != '\\':
				out += '/'
			out += config["Training"]["callback_weights_name"]
			out += "-{epoch:02d}-{loss:.4f}.hdf5"
			callbacks += [ModelCheckpoint(out, monitor='loss', verbose=1, save_best_only=True, mode='min')]
			print("Added save callback")

		nn.Data.vals['epochs'] = int(config["Training"]["epochs"])
		nn.Data.vals['batch_size'] = int(config["Training"]["batch_size"])

		print("Using callbacks:", callbacks)

		print("--======================--")
		print(" |   -== TRAINING ==-   | ")
		print(" | GUI WILL NOT RESPOND | ")
		print(" |     USE ONLY CMD     | ")
		print("--======================--")

		nn.train(nn.Data.model, nn.Data.vals, callbacks)

class GenerateScreen(Screen):
	def do_it(self):
		out = None
		if config["Generating"]["out_type"] == "console":
			out = sys.stdout
		elif config["Generating"]["out_type"] == "screen":
			def flush_call(fsio):
				pass
			out = FlushableStringIO(flush_call=flush_call)
		elif config["Generating"]["out_type"] == "file":
			out = open(config["Generating"]["out_file"])
		else:
			self.ids['feedback'].text = "Cannot find file"
			return

		diversities = []
		raw_diversities = config["Generating"]["diversities"].split(',')
		for raw_div in raw_diversities:
			diversities += [float(raw_div)]

		seed = ""
		raw_seed = config["Generating"]["seed"]
		for c in raw_seed:
			if c in nn.Data.vals['char_indx'].keys():
				seed += c

		nn.generate(nn.Data.model, nn.Data.vals, out,
			seed = seed,
			diversities = diversities,
			length = int(config["Generating"]["length"]),
			infout = False
		)

		if config["Generating"]["out_type"] == "screen":
			self.ids['feedback'].text = seed + "|" + out.getvalue()
			out.close()
		elif config["Generating"]["out_type"] == "file":
			out.close()

class ActionScreen(Screen):
	def train(self):
		pass

	def generate(self):
		pass

	def write(self):
		pass

class SettingsScreen(Screen):
	init_settings = BooleanProperty(False)

	def load_settings(self):
		if self.init_settings:
			return
		s = self.ids['sets']
		s.add_json_panel('Model', config, 'gui/config_model.json')
		s.add_json_panel('Training', config, 'gui/config_training.json')
		s.add_json_panel('Generating', config, 'gui/config_generating.json')
		s.add_json_panel('Writing', config, 'gui/config_writing.json')

		self.init_settings = True
		return s

class MainApp(App):
	pass

def ca(b):
	pass

if __name__ == '__main__':
	MainApp().run()
	# nn.trainNetwork("bnw/bnw.data.json", "bnw2", "bnw2/weights-improvement-18-1.7800.hdf5")
	# nn.generateModelText("bnw/bnw.data.json", "bnw2/weights-improvement-18-1.7800.hdf5")
