__version__ = 1.0

from kivy.uix.tabbedpanel import TabbedPanel
from kivy.lang import Builder
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.properties import StringProperty

class InternetPopup(Popup):
	pass



class LocalFilePopup(Popup):
	pass




class RootWidget(TabbedPanel):
    
    def internet_popup(self, *args):
    	InternetPopup().open()

    def local_file_popup(self, *args):
    	LocalFilePopup().open()

class DssApp(App):
    def build(self):
        return RootWidget()


if __name__ == '__main__':
    DssApp().run()
