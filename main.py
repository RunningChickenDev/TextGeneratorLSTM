from kivy.app import App
from kivy.properties import StringProperty
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock, mainthread
import neuralnetwork as nn
import os
from kivy.uix.checkbox import CheckBox
import subprocess
from threading import Thread

# class FileSelectorToInput(FloatLayout):
# 	text_input = ObjectProperty(None)
#
# 	def use(self, path, filename):
# 		text_input.text = os.path.join(path, filename[0])

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

		thread = Thread(target = nn.generateModelText, args=(metadata, weights, self.perCharCallback,))
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
