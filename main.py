__version__ = 1.0

from kivy.uix.tabbedpanel import TabbedPanel
from kivy.lang import Builder
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.properties import StringProperty
from kivy.properties import ObjectProperty
from kivy.uix.checkbox import CheckBox

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
