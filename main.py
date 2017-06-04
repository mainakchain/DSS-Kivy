__version__ = 1.0

import pandas as pd

from kivy.uix.tabbedpanel import TabbedPanel
from kivy.lang import Builder
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.properties import StringProperty
from kivy.properties import ListProperty
from kivy.properties import DictProperty
from kivy.uix.checkbox import CheckBox
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.core.window import Window
Window.clearcolor = (78/255., 208/255., 155/255., 1)

import matplotlib
matplotlib.use("module://kivy.garden.matplotlib.backend_kivyagg")
from kivy.garden.matplotlib import FigureCanvasKivy, FigureCanvasKivyAgg

from matplotlib import pyplot as plt
import seaborn as sns


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
	big_dict = DictProperty()
	empty_big_dict = DictProperty()
	column_name = ListProperty('None')
	
	def __init__(self, **kwargs):
		super(RootWidget, self).__init__(**kwargs)

	def ping(self, *args):
		# self.big_dict[column] = value
		print args[0]
		print args[1]
		self.big_dict[args[0]][1] = args[1]
		print self.big_dict[args[0]][0]

	def import_dataset(self):

		if self.file_name.endswith('zip'):
			# zf = zipfile.ZipFile(self.file_name)
			self.data = pd.read_csv(self.file_name, compression='zip', sep=',', quotechar='"')
		else:
			self.data = pd.read_csv(self.file_name)
		self.column_names = list(self.data)
		# print self.column_names
		# self.display_drop_section()
		self.ids.layout_content.clear_widgets()
		self.ids.display_info.text = str(self.data.describe())
		self.ids.update_status.text ='Dataset imported successfully!'
		print type(self.data.head())
		
		for sl, column in enumerate(self.column_names):
			label = Label(text=str(sl+1), size_hint=(0.2,1), pos_hint={'top': 0.5 + self.size_hint[1]/2})
			checkbox = CheckBox(text=column)
			checkbox.bind(active=self.ping)
			space = BoxLayout(size_hint=(0.4, 1)) 
			layout = self.ids.layout_content
			name = Label(text=column)
			self.big_dict[checkbox] = [column, False]
			layout.add_widget(label)
			layout.add_widget(space)
			layout.add_widget(checkbox)
			layout.add_widget(name)


	def display_drop_section(self):
		for sl, column in enumerate(self.column_names):
			label = Label(text=str(sl+1), size_hint=(0.2,1), pos_hint={'top': 0.5 + self.size_hint[1]/2})
			checkbox = CheckBox(text=column)
			checkbox.bind(active=self.ping)
			space = BoxLayout(size_hint=(0.4, 1)) 
			print 'done'
			layout = self.ids.layout_content
			name = Label(text=column)
			layout.add_widget(label)
			layout.add_widget(space)
			layout.add_widget(checkbox)
			layout.add_widget(name)

	def optimize(self, *args):
		graph_display = self.ids.graph_display
		sns.set_palette('colorblind')
		sns.countplot(data=self.data, x="survived", hue="pclass")
		print plt.gcf().axes
		graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))
		print "done"



	def drop_columns(self, *args):

		self.ids.update_status.text ='Columns dropped successfully!'

		for checkbox in self.big_dict:
			if self.big_dict[checkbox][1]:
				self.data.drop(self.big_dict[checkbox][0], axis=1)
				self.column_names.remove(self.big_dict[checkbox][0])
		# self.display_drop_section()
		self.ids.layout_content.clear_widgets()
		self.big_dict = self.empty_big_dict
		for sl, column in enumerate(self.column_names):
			label = Label(text=str(sl+1), size_hint=(0.2,1), pos_hint={'top': 0.5 + self.size_hint[1]/2})
			checkbox = CheckBox(text=column)
			checkbox.bind(active=self.ping)
			self.big_dict[checkbox] = [column, False]
			space = BoxLayout(size_hint=(0.4, 1)) 
			layout = self.ids.layout_content
			name = Label(text=column)
			layout.add_widget(label)
			layout.add_widget(space)
			layout.add_widget(checkbox)
			layout.add_widget(name)

		self.column_name = self.column_names
		self.ids.display_info.text = str(self.data.describe())

	def dropDown(self, *args):
		dropdown = DropDown()
		for names in self.column_name:
			btn = Button(text=names, size_hint_y=None, height=44)
			btn.bind(on_release=lambda btn: dropdown.select(btn.text))
			dropdown.add_widget(btn)
		mainbutton=self.ids.predict_mainbutton
		mainbutton.bind(on_release=dropdown.open)
		dropdown.bind(on_select=lambda instance, x: setattr(mainbutton, 'text', x))
		layout = self.ids.layout_dropdown


	def internet_popup(self, *args):
		internet = InternetPopup(self)
		internet.open()

	def local_file_popup(self, *args):
		local = LocalFilePopup(self)
		local.open()

	def optimize_SVM(self, args):

		pass
		

	



class DssApp(App):
    def build(self):
        return RootWidget()


if __name__ == '__main__':
    DssApp().run()
