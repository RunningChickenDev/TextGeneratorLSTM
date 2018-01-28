from kivy.app import App
from kivy.properties import StringProperty
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen
import neuralnetwork as nn
import os
from kivy.uix.checkbox import CheckBox
import subprocess

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

	def perCharCallback(c):
		if Generator.instance == None:
			Generator.instance = Generator()
		Generator.instance.ids["text_output"].insert_text(c)

	def generate(self):
		Generator.text = ""
		nn.generateText("bettertext.txt", "weights-improvement-20-2.8266.hdf5", Generator.perCharCallback)

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
	# nn.trainNetwork("bnw/bnw.data.json", "bnw2")
