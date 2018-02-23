from kivy.app import App
from kivy.properties import StringProperty, BooleanProperty, NumericProperty
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock, mainthread
from keras.callbacks import LambdaCallback, ModelCheckpoint
from kivy.config import ConfigParser
from kivy.uix.settings import Settings
import json
import os
import betternn as nn
import configparser as cfg

class ActionScreen(Screen):
	def load_model(self):
		pass

	def train(self):
		pass

	def generate(self):
		pass

	def write(self):
		pass

class SettingsScreen(Screen):
	def load_settings(self):
		config = ConfigParser()
		config.read('gui/config.ini')

		s = self.ids['sets']
		s.add_json_panel('Model', config, 'gui/config_model.json')
		s.add_json_panel('Training', config, 'gui/config_training.json')
		s.add_json_panel('Generating', config, 'gui/config_generating.json')
		s.add_json_panel('Writing', config, 'gui/config_writing.json')
		return s

class MainApp(App):
	pass

def ca(b):
	pass

if __name__ == '__main__':
	MainApp().run()
	# nn.trainNetwork("bnw/bnw.data.json", "bnw2", "bnw2/weights-improvement-18-1.7800.hdf5")
	# nn.generateModelText("bnw/bnw.data.json", "bnw2/weights-improvement-18-1.7800.hdf5")
