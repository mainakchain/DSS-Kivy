__version__ = 1.0

import pandas as pd

from kivy.uix.tabbedpanel import TabbedPanel
from kivy.lang import Builder
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.properties import StringProperty
from kivy.properties import ObjectProperty
from kivy.uix.checkbox import CheckBox
from kivy.uix.label import Label

import matplotlib
matplotlib.use("module://kivy.garden.matplotlib.backend_kivy")
from kivy.garden.matplotlib import FigureCanvasKivyAgg

from kivy.garden.graph import Graph, MeshLinePlot


class InternetPopup(Popup):
	
	def __init__(self, root, **kwargs):
		super(InternetPopup, self).__init__(**kwargs)
		self.root = root

	def send_file_name(self, *args):

		self.root.file_name = self.ids.url.text
		self.dismiss()


class LocalFilePopup(Popup):
	def __init__(self, root, **kwargs):
		super(LocalFilePopup, self).__init__(**kwargs)
		self.root = root


	def select(self, *args):
		self.root.file_name = args[1][0]
		self.dismiss()

class RootWidget(TabbedPanel):

	file_name = StringProperty('None')
	def __init__(self, **kwargs):
		super(RootWidget, self).__init__(**kwargs)

	def import_dataset(self):
		self.data = pd.read_csv(self.file_name)
		self.column_names = list(self.data)
		print self.column_names

		for sl, column in enumerate(self.column_names):
			label = Label(text=str(sl+1), size_hint=(0.2,1), pos_hint={'top': 0.5 + self.size_hint[1]/2})
			checkbox = CheckBox(text=column)
			space = BoxLayout(size_hint=(0.4, 1)) 
			layout = self.ids.layout_content
			name = Label(text=column)
			layout.add_widget(label)
			layout.add_widget(space)
			layout.add_widget(checkbox)
			layout.add_widget(name)

	def internet_popup(self, *args):
		internet = InternetPopup(self)
		internet.open()

	def local_file_popup(self, *args):
		local = LocalFilePopup(self)
		local.open()

	def optimize_SVM(self):
        # v_svm_c = StringProperty('')
        # v_svm_kernel = StringProperty('')
        # v_svm_degree = StringProperty('')
        # v_svm_gamma = StringProperty('')
        # v_svm_coef0 = StringProperty('')
        # v_svm_tol = StringProperty('')
		pass
	def optimize(self):
		pass


class DssApp(App):
    def build(self):
        return RootWidget()


if __name__ == '__main__':
    DssApp().run()
