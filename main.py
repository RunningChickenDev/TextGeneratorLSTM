from kivy.app import App
from kivy.properties import StringProperty
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen
import neuralnetwork as nn

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

if __name__ == '__main__':
	# nn.generateText("bettertext.txt", "weights-improvement-20-2.8266.hdf5")
	# MainApp().run()
	nn.trainModel("dataset_en.txt", "en")
