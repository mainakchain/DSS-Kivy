from kivy.uix.tabbedpanel import TabbedPanel
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.properties import StringProperty

class InternetPopup(Popup):
	text = StringProperty()


class RootWidget(TabbedPanel):
    
    def internet_popup(self, *args):
    	InternetPopup().open()


class DssApp(App):
    def build(self):
        return RootWidget()


if __name__ == '__main__':
    DssApp().run()
