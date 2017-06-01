from kivy.uix.tabbedpanel import TabbedPanel
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.properties import StringProperty
from kivy.properties import ObjectProperty
from kivy.uix.checkbox import CheckBox

class InternetPopup(Popup):
	pass



class LocalFilePopup(Popup):
	pass


# class CheckBoxes(CheckBox):
#   pass



class RootWidget(TabbedPanel):

    
    
    def internet_popup(self, *args):
        InternetPopup().open()

    def local_file_popup(self, *args):
        LocalFilePopup().open()

    # def check_boxes(self, *args):
    #   CheckBoxes().active()
    

class DssApp(App):
    def build(self):
        return RootWidget()


if __name__ == '__main__':
    DssApp().run()
