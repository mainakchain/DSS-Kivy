__version__ = 1.0

from kivy.uix.tabbedpanel import TabbedPanel
from kivy.lang import Builder
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout



class RootWidget(TabbedPanel):
    pass

class DssApp(App):
    def build(self):
        return RootWidget()

if __name__ == '__main__':
    DssApp().run()
