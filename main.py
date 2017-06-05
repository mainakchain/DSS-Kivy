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
from kivy.properties import NumericProperty
from kivy.properties import ObjectProperty
from kivy.uix.checkbox import CheckBox
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.uix.scrollview import ScrollView
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
	column_names = ''
	value = NumericProperty()
	
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

	def draw_graph(self, *args):
		graph_display = self.ids.graph_display
		sns.set_palette('colorblind')
		sns.countplot(data=self.data, x="survived", hue="pclass")
		print plt.gcf().axes
		graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))
		print "done"

	def predict(self, *args):
		predict_graph_display = self.ids.predict_graph
		sns.set_palette('colorblind')
		sns.countplot(data=self.data, x="survived", hue="pclass")
		print plt.gcf().axes
		predict_graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))
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

	def dropDown(self, lists, *args):				#ID will be mainbutton id
		dropdown = DropDown()
		print args[0]
		print lists
		for names in lists:
			print names
			btn = Button(text=names, size_hint_y=None, height=25)
			btn.bind(on_release=lambda btn: dropdown.select(btn.text))
			dropdown.add_widget(btn)
		args[0].bind(on_release=dropdown.open)
		dropdown.bind(on_select=lambda instance, x: setattr(args[0] ,'text', x))
		# scroll = ScrollView(size_hint=(1, None), do_scroll_y=True, do_scroll_x=False)
		# scroll.add_widget(self.ids.layout_dropdown)

	def internet_popup(self, *args):
		internet = InternetPopup(self)
		internet.open()

	def local_file_popup(self, *args):
		local = LocalFilePopup(self)
		local.open()

	def optimize_parameters(self, value):
		layout = self.ids.layout_optimize_parameters

		if value==1:
			c_label = Label(text='C')
			c_lower = TextInput(multiline=False,
                                   size_hint=(.5, None), height=30, width=10, hint_text='lower c value')
			c_upper = TextInput(multiline=False,
                                   size_hint=(.5, None), height=30, width=10, hint_text='upper c value')
			layout.add_widget(c_label)
			layout.add_widget(c_lower, )
			layout.add_widget(c_upper, )

			tol_label = Label(text='tol')
			tol_lower = TextInput(multiline=False,
                                   size_hint=(.5, None), height=30, width=10, hint_text='lower tol value')
			tol_upper = TextInput(multiline=False,
                                   size_hint=(.5, None), height=30, width=10, hint_text='upper tol value')
			layout.add_widget(tol_label)
			layout.add_widget(tol_lower)
			layout.add_widget(tol_upper)

			degree_label = Label(text='degree')
			degree_lower = TextInput(multiline=False,
                                   size_hint=(.5, None), height=30, width=10, hint_text='lower degree value')
			degree_upper = TextInput(multiline=False,
                                   size_hint=(.5, None), height=30, width=10, hint_text='upper degree value')
			layout.add_widget(degree_label)
			layout.add_widget(degree_lower)
			layout.add_widget(degree_upper)

			gamma_label = Label(text='gamma')
			gamma_lower = TextInput(multiline=False,
                                   size_hint=(.5, None), height=30, width=10, hint_text='lower gamma value')
			gamma_upper = TextInput(multiline=False,
                                   size_hint=(.5, None), height=30, width=10, hint_text='upper gamma value')
			layout.add_widget(gamma_label)
			layout.add_widget(gamma_lower)
			layout.add_widget(gamma_upper)

			coef0_label = Label(text='coef0')
			coef0_lower = TextInput(multiline=False,
                                   size_hint=(.5, None), height=30, width=10, hint_text='lower coef0 value')
			coef0_upper = TextInput(multiline=False,
                                   size_hint=(.5, None), height=30, width=10, hint_text='upper coef0 value')
			layout.add_widget(coef0_label)
			layout.add_widget(coef0_lower)
			layout.add_widget(coef0_upper)

			kernel_label = Label(text = 'kernel')
			kernel_mainbutton = Button(text='Choose kernel')
			kernel_mainbutton.bind(on_press=lambda x:self.dropDown(self.column_names, kernel_mainbutton))
			layout.add_widget(kernel_label)
			layout.add_widget(kernel_mainbutton)



class DssApp(App):
    def build(self):
        return RootWidget()


if __name__ == '__main__':
    DssApp().run()