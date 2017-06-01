__version__ = 1.0

from kivy.uix.tabbedpanel import TabbedPanel
from kivy.lang import Builder
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.properties import StringProperty

import matplotlib
matplotlib.use("module://kivy.garden.matplotlib.backend_kivy")
from kivy.garden.matplotlib import FigureCanvasKivyAgg

from kivy.garden.graph import Graph, MeshLinePlot


class InternetPopup(Popup):
	pass



class LocalFilePopup(Popup):
	pass


class RootWidget(TabbedPanel):
    
    def internet_popup(self, *args):
    	InternetPopup().open()

    def local_file_popup(self, *args):
    	LocalFilePopup().open()

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
