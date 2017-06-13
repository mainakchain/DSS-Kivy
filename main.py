__version__ = 1.0

import pandas as pd

from kivy.uix.tabbedpanel import TabbedPanel
from kivy.lang import Builder
from kivy.app import App
from kivy.base import runTouchApp
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
from kivy.uix.spinner import Spinner
from kivy.uix.dropdown import DropDown
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window


				 
from sklearn.svm import SVC # SVM
from sklearn.ensemble import RandomForestClassifier #RandomForest
from sklearn.neighbors import KNeighborsClassifier #KNeighbors
from sklearn.neural_network import MLPClassifier #ANN
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score




Window.clearcolor = (78/255., 208/255., 155/255., 1)

Builder.load_string('''
<SpinnerOption>:
    size_hint_y: None
    height: 30
''')

from kivy.config import Config
Config.set('graphics', 'fullscreen', 'auto')

import matplotlib
matplotlib.use("module://kivy.garden.matplotlib.backend_kivyagg")
from kivy.garden.matplotlib import FigureCanvasKivy, FigureCanvasKivyAgg

from matplotlib import pyplot as plt
import seaborn as sns


class InternetPopup(Popup):	
	

	def __init__(self, root, **kwargs):
		super(InternetPopup, self).__init__(**kwargs)
		self.root = root
		self.auto_dismiss = True

	def send_file_name(self, value, *args):

		self.root.file_name = self.ids.url.text
		self.dismiss()


class LocalFilePopup(Popup):
	
	def __init__(self, root, **kwargs):
		super(LocalFilePopup, self).__init__(**kwargs)
		self.root = root
		self.auto_dismiss = True


	def select(self, *args):
			self.root.file_name = args[1][0]
			self.dismiss()

class LocalTestFilePopup(Popup):
	test_filename = StringProperty('None')
	

	def __init__(self, root, **kwargs):
			super(LocalTestFilePopup, self).__init__(**kwargs)
			self.root = root
			self.auto_dismiss = True

	def select(self, *args):
		self.root.test_file_name = args[1][0]
		self.test_filename = args[1][0]

	def call_import(self, *args):
		self.root.import_test_dataset()
		self.dismiss()

class ManualInputPopup(Popup):
	

	def __init__(self, root, **kwargs):
			super(ManualInputPopup, self).__init__(**kwargs)
			self.root = root
			self.auto_dismiss = True

	def populate(self, *args):
		layout = self.ids.layout_manual_test_data

		for col in self.root.column_names:
			layout.add_widget(Label(text=col))
			layout.add_widget(TextInput(multiline=False,size_hint_y=None,height=30))

	def done(self, *args):
		if (self.root.test_data != []):
			self.root.ids.predict_update_status.text = 'Test Data imported successfully! '
		else:
			self.root.ids.predict_update_status.text = 'Test Data empty! Please input data!'
		self.dismiss()


class RootWidget(TabbedPanel):

	file_name = StringProperty('None')
	test_file_name = StringProperty('None')
	big_dict = DictProperty()
	empty_big_dict = DictProperty()
	column_names = []
	number_of_columns = NumericProperty(len(column_names))
	value = NumericProperty()
	columns = ListProperty(column_names)
	data = []
	test_data = []
	checkbox1 = CheckBox(group='algo_selection')
	model = ObjectProperty()

	params = DictProperty()
	
	def __init__(self, **kwargs):
		super(RootWidget, self).__init__(**kwargs)
		Window.size = (1300, 700)

	def ping(self, *args):
		# self.big_dict[column] = value
		print args[0]
		print args[1]
		self.big_dict[args[0]][1] = args[1]
		print self.big_dict[args[0]][0]

	def clean(self, column):
		i = 0
		seperator_list = []
		local_data = self.data[column]
		new_local_data = []
		for entry in local_data:
			if entry:
				if entry not in seperator_list:
					seperator_list.append(entry)
					new_local_data.append(i)
					i += 1
				else :
					new_local_data.append(seperator_list.index(entry))
			else:
				new_local_data.append(0)
		self.data[column] = new_local_data

	def import_dataset(self):

		if self.file_name.endswith('zip'):
			# zf = zipfile.ZipFile(self.file_name)
			self.data = pd.read_csv(self.file_name, compression='zip', sep=',', quotechar='"')
		else:
			self.data = pd.read_csv(self.file_name)
		self.column_names = list(self.data)
		self.number_of_columns = len(self.column_names)
		self.columns = self.column_names
		for val, column in enumerate(self.data.dtypes):
			if column == 'object':
				self.clean(self.data.columns[val])


		self.ids.layout_content.clear_widgets()
		# self.ids.set_features.clear_widgets()
		self.ids.display_info.text = str(self.data.describe())
		self.ids.update_status.text ='Dataset imported successfully!'
		# scroll_layout = self.ids.set_features
		print type(self.data.head())
		
		for sl, column in enumerate(self.column_names):
			checkbox = CheckBox(text=column)
			checkbox.bind(active=self.ping)
			# space = BoxLayout(size_hint=(0.4, 1)) 
			layout = self.ids.layout_content
			name = Label(text=column)
			self.big_dict[checkbox] = [column, False]
			# layout.add_widget(space) 	
			layout.add_widget(checkbox)
			layout.add_widget(name)
		for column in self.column_names:
			lab = Label(text=column, size_hint_x=None, width=100)
			ent = TextInput(size_hint_x=None, width=200)
			# scroll_layout.add_widget(lab)
			# scroll_layout.add_widget(ent)

	def import_test_dataset(self):
		if self.test_file_name.endswith('zip'):
			self.test_data = pd.read_csv(self.test_file_name, compression='zip', sep=',', quotechar='"')
		else:
			self.test_data = pd.read_csv(self.test_file_name)

		self.ids.predict_update_status.text = 'Test Data imported successfully! '
		

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

	def draw_graph(self, X, Y, type_graph, hue):

		
		graph_display = self.ids.graph_display
		graph_display.clear_widgets()
		graph = type_graph.text
		sns.set_palette('colorblind')
		
		if graph == 'Count Plot':

			sns.countplot(data=self.data, x=X.text, hue=hue.text)
			plt.xticks(rotation='vertical')
			print plt.gcf().axes
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))

		elif graph == 'Pair Plot':
			sns.pairplot(self.data, hue=hue.text, size=6, x_vars=X.text, y_vars=Y.text )
			print plt.gcf().axes
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))

		elif graph == 'Factor Plot':
			sns.factorplot(data=self.data, x=X.text, y=Y.text, col=hue.text)
			# plt.x_ticks()
			print plt.gcf().axes
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))

		elif graph == 'Dist Plot':
			g = sns.FacetGrid(self.data, col=hue.text)  
			g.map(sns.distplot, X.text)
			print plt.gcf().axes
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))

		elif graph == 'Scatter Plot':
			g = sns.FacetGrid(self.data, col=hue.text)  
			g.map(plt.scatter, X.text, Y.text)
			print plt.gcf().axes
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))

		elif graph == 'Reg Plot':
			g = sns.FacetGrid(self.data, col=hue.text)  
			g.map(sns.regplot, X.text, Y.text)
			print plt.gcf().axes
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))

		elif graph == 'Kde Plot':
			g = sns.FacetGrid(self.data, col=hue.text, row="survived")  
			g.map(sns.kdeplot, X.text, Y.text)
 			print plt.gcf().axes
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))

		elif graph == 'Joint Plot':
			sns.jointplot(X.text, Y.text, data=self.data, kind='kde')
			print plt.gcf().axes
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))

		elif graph == 'Violin Plot':
			sns.violinplot(x=X.text, y=Y.text, data=self.data)
			print plt.gcf().axes
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))
 
	def predict(self, *args):
		predict_graph_display = self.ids.predict_graph
		sns.set_palette('colorblind')
		sns.countplot(data=self.data, x="survived", hue="pclass")
		print plt.gcf().axes
		predict_graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))
		print "done"

	def cross_validate(self, *args):
		classifier = self.ids.choose_classifier.text
		roc_auc = 'Multiclass label'
		precision = 'Multiclass label'
		recall = 'Multiclass label'
		print self.model
		self.model.fit(self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text])
		accuracy = cross_val_score(self.model, self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text], cv=10, scoring='accuracy').mean()
		if len(self.data[self.ids.predict_dropdown_choose_parameter.text].unique()) < 3:
			precision = cross_val_score(self.model, self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text], cv=10, scoring='average_precision').mean()
		f1_score = cross_val_score(self.model, self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text], cv=10, scoring='f1_weighted').mean()
		if len(self.data[self.ids.predict_dropdown_choose_parameter.text].unique()) < 3:
			recall = cross_val_score(self.model, self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text], cv=10, scoring='recall').mean()
		if len(self.data[self.ids.predict_dropdown_choose_parameter.text].unique()) < 3:
			roc_auc = cross_val_score(self.model, self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text], cv=10, scoring='roc_auc').mean()
		self.ids.accuracy.text = str(accuracy)
		self.ids.precision.text = str(precision)
		self.ids.f1.text = str(f1_score)
		self.ids.recall.text = str(recall)
		self.ids.auc_roc.text = str(roc_auc)

	def prediction(self):
		classifier_type = self.ids.choose_classifier.text
		params = self.params
		print params
		if classifier_type == 'SVM':
			C = float(params['C'])
			kernel = params['kernel']
			degree = int(params['degree'])
			# gamma = float(params['gamma'])
			tol = float(params['tol'])
			coef0 = float(params['coef0'])

			self.model = SVC(C=C, kernel=kernel, degree=degree, tol=tol, coef0=coef0)

		if classifier_type == 'ANN':
			hidden_layer_sizes = tuple(params['hidden_layer_sizes'])
			max_iter = int(params['max_iter'])
			# activation = params['activation']
			solver = params['solver']
			learning_rate = params['learning_rate']
			# momentum = params['momentum']

			self.model = MLPClassifier()

		if classifier_type == 'Random Forest':
			n_estimators = int(params['n_estimators'])
			min_samples_leaf = float(params['min_samples_leaf'])
			# max_depth = int(params['max_depth'])
			min_samples_split = float(params['min_samples_split'])
			min_weight_fraction_leaf = float(params['min_weight_fraction_leaf'])
			# max_leaf_nodes = int(params['max_leaf_nodes'])

			self.model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf)


		if classifier_type == 'KNN':
			n_neighbors = int(params['n_neighbors'])
			# p = int(params['p'])
			leaf_size = int(params['leaf_size'])
			n_jobs = int(params['n_jobs'])
			algorithm = params['algorithm']
			weights = params['weights']

			self.model = KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size, n_jobs=n_jobs, algorithm=algorithm, weights=weights)
		print self.model
		print classifier_type



	def drop_columns(self, *args):

		self.ids.update_status.text ='Columns dropped successfully!'
		for checkbox in self.big_dict:
			if self.big_dict[checkbox][1]:
				self.data.drop(self.big_dict[checkbox][0], axis=1)
				self.column_names.remove(self.big_dict[checkbox][0])
		# self.display_drop_section()
		self.ids.layout_content.clear_widgets()
		# self.ids.set_features.clear_widgets()
		self.number_of_columns = len(self.column_names)
		self.big_dict = self.empty_big_dict
		for sl, column in enumerate(self.column_names):
			checkbox = CheckBox(text=column)
			checkbox.bind(active=self.ping)
			self.big_dict[checkbox] = [column, False]
			layout = self.ids.layout_content
			name = Label(text=column)
			layout.add_widget(checkbox)
			layout.add_widget(name)

		for column in self.column_names:
			lab = Label(text=column, size_hint_x=None, width=100)
			ent = TextInput(size_hint_x=None, width=200)
			# self.ids.set_features.add_widget(lab)
			# self.ids.set_features.add_widget(ent)

		self.column_name = self.column_names
		self.columns = self.column_names
		self.ids.display_info.text = str(self.data.describe())

	def dropDown(self, lists, *args):
		dropdown = DropDown()
		for names in lists:
			btn = Button(text=names, size_hint_y=None, height=30)
			btn.bind(on_release=lambda btn: dropdown.select(btn.text))
			dropdown.add_widget(btn)
		args[0].bind(on_release=dropdown.open)
		dropdown.bind(on_select=lambda instance, x: setattr(args[0] ,'text', x))
		# scroll = ScrollView(size_hint=(1, None), do_scroll_y=True, do_scroll_x=False)
		# scroll.add_widget(self.ids.layout_dropdown)

	def optimize_algo_selection(self, *args):
		check = self.ids.predict_checkbox_algo
		layout = self.ids.predict_optimize_algo_selection
		if check.active == True:
			layout.clear_widgets()

			
			self.checkbox1.bind( on_press=lambda x: self.optimize_data('gridcv'),on_release=lambda x:self.populate_predict_algo_parameters())
			label1 = Label(text='GridSearchCV')
			checkbox2 = CheckBox(group='algo_selection')
			checkbox2.bind( on_press=lambda x: self.optimize_data('geneticalgo'))
			label2 = Label(text='Genetic Algorithm')
			layout.add_widget(self.checkbox1)		
			layout.add_widget(label1)
			layout.add_widget(checkbox2)
			layout.add_widget(label2)
		else:
			layout.clear_widgets()

	def populate_predict_algo_parameters(self):
		layout = self.ids.predict_optimize_algo_parameters

		if self.checkbox1.active == True:
			if self.ids.choose_classifier.text == 'SVM':
				layout.clear_widgets()
				c_label = Label(text='c', color=(1,1,1,2))
				c_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower c value')
				c_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper c value')
				layout.add_widget(c_label)
				layout.add_widget(c_lower)
				layout.add_widget(c_upper)

				tol_label = Label(text='tol', color=(1,1,1,2))
				tol_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower tol value')
				tol_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper tol value')
				layout.add_widget(tol_label)
				layout.add_widget(tol_lower)
				layout.add_widget(tol_upper)

				degree_label = Label(text='degree', color=(1,1,1,2))
				degree_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower degree value')
				degree_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper degree value')
				layout.add_widget(degree_label)
				layout.add_widget(degree_lower)
				layout.add_widget(degree_upper)

				gamma_label = Label(text='gamma', color=(1,1,1,2))
				gamma_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower gamma value')
				gamma_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper gamma value')
				layout.add_widget(gamma_label)
				layout.add_widget(gamma_lower)
				layout.add_widget(gamma_upper)

				coef0_label = Label(text='coef0', color=(1,1,1,2))
				coef0_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower coef0 value')
				coef0_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper coef0 value')
				layout.add_widget(coef0_label)
				layout.add_widget(coef0_lower)
				layout.add_widget(coef0_upper)

				kernel_label = Label(text = 'kernel', color=(1,1,1,2))
				kernel_mainbutton = Button(text='Choose kernel',size_hint=(None, None), height=30,width=140)
				kernel_mainbutton.bind(on_press=lambda x:self.dropDown(['rbf','linear','poly','sigmoid','precomputed'], kernel_mainbutton))
				layout.add_widget(kernel_label)
				layout.add_widget(kernel_mainbutton)

			if self.ids.choose_classifier.text == 'Random Forest':
				layout.clear_widgets()
				n_estimators_label = Label(text='n estimators', color=(1,1,1,2))
				n_estimators_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				n_estimators_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(n_estimators_label)
				layout.add_widget(n_estimators_lower)
				layout.add_widget(n_estimators_upper)

				min_samples_leaf_label = Label(text='min samples\n    leaf', color=(1,1,1,2))
				min_samples_leaf_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				min_samples_leaf_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(min_samples_leaf_label)
				layout.add_widget(min_samples_leaf_lower)
				layout.add_widget(min_samples_leaf_upper)

				max_depth_label = Label(text='max depth', color=(1,1,1,2))
				max_depth_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				max_depth_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(max_depth_label)
				layout.add_widget(max_depth_lower)
				layout.add_widget(max_depth_upper)

				min_samples_split_label = Label(text='min samples\n     split', color=(1,1,1,2))
				min_samples_split_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				min_samples_split_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(min_samples_split_label)
				layout.add_widget(min_samples_split_lower)
				layout.add_widget(min_samples_split_upper)

				min_weight_fraction_leaf_label = Label(text='min weight\nfraction leaf', color=(1,1,1,2))
				min_weight_fraction_leaf_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				min_weight_fraction_leaf_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(min_weight_fraction_leaf_label)
				layout.add_widget(min_weight_fraction_leaf_lower)
				layout.add_widget(min_weight_fraction_leaf_upper)

				max_leaf_nodes_label = Label(text='max leaf\n  nodes', color=(1,1,1,2))
				max_leaf_nodes_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				max_leaf_nodes_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(max_leaf_nodes_label)
				layout.add_widget(max_leaf_nodes_lower)
				layout.add_widget(max_leaf_nodes_upper)

			if self.ids.choose_classifier.text == 'KNN':
				layout.clear_widgets()
				n_neighbors_label = Label(text='n neighbors', color=(1,1,1,2))
				n_neighbors_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				n_neighbors_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(n_neighbors_label)
				layout.add_widget(n_neighbors_lower)
				layout.add_widget(n_neighbors_upper)

				p_label = Label(text='p', color=(1,1,1,2))
				p_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower p value')
				p_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper p value')
				layout.add_widget(p_label)
				layout.add_widget(p_lower)
				layout.add_widget(p_upper)

				leaf_size_label = Label(text='leaf size', color=(1,1,1,2))
				leaf_size_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				leaf_size_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(leaf_size_label)
				layout.add_widget(leaf_size_lower)
				layout.add_widget(leaf_size_upper)

				n_jobs_label = Label(text='n jobs', color=(1,1,1,2))
				n_jobs_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				n_jobs_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(n_jobs_label)
				layout.add_widget(n_jobs_lower)
				layout.add_widget(n_jobs_upper)

				algorithm_label = Label(text = 'algorithm', color=(1,1,1,2))
				algorithm_mainbutton = Button(text='Choose kernel',size_hint=(None, None), height=30,width=140)
				algorithm_mainbutton.bind(on_press=lambda x:self.dropDown(['ball tree','kd tree','brute','auto'], algorithm_mainbutton))
				layout.add_widget(algorithm_label)
				layout.add_widget(algorithm_mainbutton)

				dummy_label = Label(size_hint=(None, None), height=30,width=140)
				layout.add_widget(dummy_label)

				weights_label = Label(text = 'weights', color=(1,1,1,2))
				weights_mainbutton = Button(text='Choose kernel',size_hint=(None, None), height=30,width=140)
				weights_mainbutton.bind(on_press=lambda x:self.dropDown(['rbf','uniform', 'distance'], weights_mainbutton))
				layout.add_widget(weights_label)
				layout.add_widget(weights_mainbutton)

			if self.ids.choose_classifier.text == 'ANN':
				layout.clear_widgets()
				hidden_layer_sizes_label = Label(text='hidden layer\n   sizes', color=(1,1,1,2))
				hidden_layer_sizes_input_lower= TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				hidden_layer_sizes_input_upper= TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(hidden_layer_sizes_label)
				layout.add_widget(hidden_layer_sizes_input_lower)
				layout.add_widget(hidden_layer_sizes_input_upper)


				max_iter_label = Label(text='max iter', color=(1,1,1,2))
				max_iter_input_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				max_iter_input_upper= TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(max_iter_label)
				layout.add_widget(max_iter_input_lower)
				layout.add_widget(max_iter_input_upper)


				activation_label = Label(text = 'activation', color=(1,1,1,2))
				activation_spinner = Spinner(text = 'Select', values=['identity', 'logistic', 'tanh', 'relu'])
				layout.add_widget(activation_label)
				layout.add_widget(activation_spinner)

				dummy_label = Label(size_hint=(None, None), height=30,width=140)
				layout.add_widget(dummy_label)

				solver_label = Label(text = 'solver', color=(1,1,1,2))
				solver_spinner = Spinner(text = 'Select', values=['lbfgs', 'sgd', 'adam'])
				layout.add_widget(solver_label)
				layout.add_widget(solver_spinner)

				dummy_label = Label(size_hint=(None, None), height=30,width=140)
				layout.add_widget(dummy_label)

				learning_rate_label = Label(text = 'learning rate', color=(1,1,1,2))
				learning_rate_spinner = Spinner(text = 'Select', values= ['adaptive', 'invscaling','constant'])
				layout.add_widget(learning_rate_label)
				layout.add_widget(learning_rate_spinner)

				dummy_label = Label(size_hint=(None, None), height=30,width=140)
				layout.add_widget(dummy_label)


				# if solver_spinner.text == 'sgd':
				momentum_label = Label(text='momentum', color=(1,1,1,2))
				momentum_input_lower= TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				momentum_input_upper= TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(momentum_label)
				layout.add_widget(momentum_input_lower)
				layout.add_widget(momentum_input_upper)
		else:
			layout.clear_widgets()






	def manual_parameter_selection(self, *args):
		pass

	def optimize_data(self, name):
		
		classifier = self.ids.choose_classifier.text
		model = self.prediction()

		if name == 'gridcv':
			pass
			# accuracy = cross_val_score(model, self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text], cv=10, scoring='accuracy').mean()
			# precision = cross_val_score(model, self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text], cv=10, scoring='average_precision').mean()
			# f1_score = cross_val_score(model, self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text], cv=10, scoring='f1').mean()
			# recall = cross_val_score(model, self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text], cv=10, scoring='recall').mean()
			# roc_auc = cross_val_score(model, self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text], cv=10, scoring='roc_auc').mean()
			# self.ids.accuracy.text = accuracy
			# self.ids.precision.text = precision
			# self.ids.f1.text = f1_score
			# self.ids.recall.text = recall
			# self.ids.auc_roc.text = roc_auc

	def updateSubSpinner(self, text):
		self.ids.choose_classifier.text = '< Select >'

		if text == 'Tree based':
			self.ids.choose_classifier.values = ['Random Forest']
		if text == 'Non-Tree based':
			self.ids.choose_classifier.values = ['SVM','ANN','KNN']

	def internet_popup(self, *args):
		internet = InternetPopup(self)
		internet.open()

	def local_file_popup(self, *args):
		local = LocalFilePopup(self)
		local.open()

	def local_test_file_popup(self, *args):
		local_test = LocalTestFilePopup(self)
		local_test.open()

	def test_data_popup(self, *args):
		test = TestDataPopup(self)
		test.open()

	def manual_input_popup(self, *args):
		man_input = ManualInputPopup(self)
		man_input.open()

	def predict_model_parameters(self, value):
		layout = self.ids.layout_predict_parameters

		if value=='SVM':


			layout.clear_widgets()
			c_label = Label(text='C', color=(1,1,1,2))
			c_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='1.0')
			layout.add_widget(c_label)
			layout.add_widget(c_input)
			self.params['C'] = c_input.text

			tol_label = Label(text='tol', color=(1,1,1,2))
			tol_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='0.049787')
			layout.add_widget(tol_label)
			layout.add_widget(tol_input)
			self.params['tol'] = tol_input.text

			degree_label = Label(text='degree', color=(1,1,1,2))
			degree_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='3')
			layout.add_widget(degree_label)
			layout.add_widget(degree_input)
			self.params['degree'] = degree_input.text

			gamma_label = Label(text='gamma', color=(1,1,1,2))
			gamma_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='auto')
			layout.add_widget(gamma_label)
			layout.add_widget(gamma_input)
			self.params['gamma'] = gamma_input.text

			coef0_label = Label(text='coef0', color=(1,1,1,2))
			coef0_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='0.0')
			layout.add_widget(coef0_label)
			layout.add_widget(coef0_input)
			self.params['coef0'] = coef0_input.text

			kernel_label = Label(text = 'kernel', color=(1,1,1,2))
			kernel_spinner = Spinner(text='rbf',values=['linear','poly','sigmoid','precomputed','rbf'])
			layout.add_widget(kernel_label)
			layout.add_widget(kernel_spinner)
			self.params['kernel'] = kernel_spinner.text

			if self.checkbox1.active == True:
				self.populate_predict_algo_parameters()

		if value=='Random Forest' :

			layout.clear_widgets()
			n_estimators_label = Label(text='n estimators', color=(1,1,1,2),size=self.parent.size)
			n_estimators_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='10')
			layout.add_widget(n_estimators_label)
			layout.add_widget(n_estimators_input)
			self.params['n_estimators'] = n_estimators_input.text



			min_samples_leaf_label = Label(text='min samples\n    leaf', color=(1,1,1,2),size=self.parent.size)
			min_samples_leaf_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='0.5')
			layout.add_widget(min_samples_leaf_label)
			layout.add_widget(min_samples_leaf_input)
			self.params['min_samples_leaf'] = min_samples_leaf_input.text


			max_depth_label = Label(text='max depth', color=(1,1,1,2),size=self.parent.size)
			max_depth_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='None')
			layout.add_widget(max_depth_label)
			layout.add_widget(max_depth_input)
			self.params['max_depth'] = max_depth_input.text

			min_samples_split_label = Label(text='min samples\n     split', color=(1,1,1,2),size=self.parent.size)
			min_samples_split_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='2')
			layout.add_widget(min_samples_split_label)
			layout.add_widget(min_samples_split_input)
			self.params['min_samples_split'] = min_samples_split_input.text

			min_weight_fraction_leaf_label = Label(text='min weight\nfraction leaf', color=(1,1,1,2),size=self.parent.size)
			min_weight_fraction_leaf_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='0.0')
			layout.add_widget(min_weight_fraction_leaf_label)
			layout.add_widget(min_weight_fraction_leaf_input)
			self.params['min_weight_fraction_leaf'] = min_weight_fraction_leaf_input.text

			max_leaf_nodes_label = Label(text='max leaf\n  nodes', color=(1,1,1,2),size=self.parent.size)
			max_leaf_nodes_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='None')
			layout.add_widget(max_leaf_nodes_label)
			layout.add_widget(max_leaf_nodes_input)
			self.params['max_leaf_nodes'] = max_leaf_nodes_input.text

			if self.checkbox1.active == True:
				self.populate_predict_algo_parameters()



		if value=='KNN':
			layout.clear_widgets()
			n_neighbors_label = Label(text='n neighbors', color=(1,1,1,2))
			n_neighbors_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='5')
			layout.add_widget(n_neighbors_label)
			layout.add_widget(n_neighbors_input)
			self.params['n_neighbors'] = n_neighbors_input.text

			p_label = Label(text='p', color=(1,1,1,2))
			p_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='0.0')
			layout.add_widget(p_label)
			layout.add_widget(p_input)
			self.params['p'] = p_input.text

			leaf_size_label = Label(text='leaf size', color=(1,1,1,2))
			leaf_size_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='30')
			layout.add_widget(leaf_size_label)
			layout.add_widget(leaf_size_input)
			self.params['leaf_size'] = leaf_size_input.text

			n_jobs_label = Label(text='n jobs', color=(1,1,1,2))
			n_jobs_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='1')
			layout.add_widget(n_jobs_label)
			layout.add_widget(n_jobs_input)
			self.params['n_jobs'] = n_jobs_input.text

			algorithm_label = Label(text = 'algorithm', color=(1,1,1,2))
			algorithm_spinner = Spinner(text='auto',values=['ball tree','kd tree','brute','auto'])
			layout.add_widget(algorithm_label)
			layout.add_widget(algorithm_spinner)
			self.params['algorithm'] = algorithm_spinner.text


			weights_label = Label(text = 'weights', color=(1,1,1,2))
			weights_spinner = Spinner(text='uniform', values=['rbf', 'distance','uniform'])
			layout.add_widget(weights_label)
			layout.add_widget(weights_spinner)
			self.params['weights'] = weights_spinner.text


			if self.checkbox1.active == True:
				self.populate_predict_algo_parameters()

		if value == 'ANN':
			layout.clear_widgets()
			hidden_layer_sizes_label = Label(text='hidden layer\n   sizes', color=(1,1,1,2))
			hidden_layer_sizes_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='100')
			layout.add_widget(hidden_layer_sizes_label)
			layout.add_widget(hidden_layer_sizes_input)
			self.params['hidden_layer_sizes'] = hidden_layer_sizes_input.text


			max_iter_label = Label(text='max iter', color=(1,1,1,2))
			max_iter_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='200')
			layout.add_widget(max_iter_label)
			layout.add_widget(max_iter_input)
			self.params['max_iter'] = max_iter_input.text


			activation_label = Label(text = 'activation', color=(1,1,1,2))
			activation_spinner = Spinner(text = 'relu', values=['identity', 'logistic', 'tanh', 'relu'])
			layout.add_widget(activation_label)
			layout.add_widget(activation_spinner)
			self.params['n_estimators'] = activation_spinner.text


			solver_label = Label(text = 'solver', color=(1,1,1,2))
			solver_spinner = Spinner(text = 'adam', values=['lbfgs', 'sgd', 'adam'])
			layout.add_widget(solver_label)
			layout.add_widget(solver_spinner)
			self.params['solver'] = solver_spinner.text



			learning_rate_label = Label(text = 'learning rate', color=(1,1,1,2))
			learning_rate_spinner = Spinner(text = 'constant', values= ['adaptive', 'invscaling','constant'])
			layout.add_widget(learning_rate_label)
			layout.add_widget(learning_rate_spinner)
			self.params['learning_rate'] = learning_rate_spinner.text



			# if solver_spinner.text == 'sgd':
			momentum_label = Label(text='momentum', color=(1,1,1,2))
			momentum_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='.9')
			layout.add_widget(momentum_label)
			layout.add_widget(momentum_input)
			
			if self.checkbox1.active == True:
				self.populate_predict_algo_parameters()
				

	# def predict_model_parameters(self, value):
	# 	layout = self.ids.layout_optimize_parameters

	# 	if value=='SVM':
	# 		layout.clear_widgets()
	# 		c_label = Label(text='C', color=(1,1,1,2))
	# 		c_lower = TextInput(multiline=False,
 #                                   size_hint=(None, None), height=30,width=140, hint_text='lower c value')
	# 		c_upper = TextInput(multiline=False,
 #                                   size_hint=(None, None), height=30,width=140, hint_text='upper c value')
	# 		layout.add_widget(c_label)
	# 		layout.add_widget(c_lower)
	# 		layout.add_widget(c_upper)

	# 		tol_label = Label(text='tol', color=(1,1,1,2))
	# 		tol_lower = TextInput(multiline=False,
 #                                   size_hint=(None, None), height=30,width=140, hint_text='lower tol value')
	# 		tol_upper = TextInput(multiline=False,
 #                                   size_hint=(None, None), height=30,width=140, hint_text='upper tol value')
	# 		layout.add_widget(tol_label)
	# 		layout.add_widget(tol_lower)
	# 		layout.add_widget(tol_upper)

	# 		degree_label = Label(text='degree', color=(1,1,1,2))
	# 		degree_lower = TextInput(multiline=False,
 #                                   size_hint=(None, None), height=30,width=140, hint_text='lower degree value')
	# 		degree_upper = TextInput(multiline=False,
 #                                   size_hint=(None, None), height=30,width=140, hint_text='upper degree value')
	# 		layout.add_widget(degree_label)
	# 		layout.add_widget(degree_lower)
	# 		layout.add_widget(degree_upper)

	# 		gamma_label = Label(text='gamma', color=(1,1,1,2))
	# 		gamma_lower = TextInput(multiline=False,
 #                                   size_hint=(None, None), height=30,width=140, hint_text='lower gamma value')
	# 		gamma_upper = TextInput(multiline=False,
 #                                   size_hint=(None, None), height=30,width=140, hint_text='upper gamma value')
	# 		layout.add_widget(gamma_label)
	# 		layout.add_widget(gamma_lower)
	# 		layout.add_widget(gamma_upper)

	# 		coef0_label = Label(text='coef0', color=(1,1,1,2))
	# 		coef0_lower = TextInput(multiline=False,
 #                                   size_hint=(None, None), height=30,width=140, hint_text='lower coef0 value')
	# 		coef0_upper = TextInput(multiline=False,
 #                                   size_hint=(None, None), height=30,width=140, hint_text='upper coef0 value')
	# 		layout.add_widget(coef0_label)
	# 		layout.add_widget(coef0_lower)
	# 		layout.add_widget(coef0_upper)

	# 		kernel_label = Label(text = 'kernel', color=(1,1,1,2))
	# 		kernel_mainbutton = Button(text='Choose kernel',size_hint=(None, None), height=30,width=140)
	# 		kernel_mainbutton.bind(on_press=lambda x:self.dropDown(['rbf','linear','poly','sigmoid','precomputed'], kernel_mainbutton))
	# 		layout.add_widget(kernel_label)
	# 		layout.add_widget(kernel_mainbutton)

	# 	if value==2 :
	# 		layout.clear_widgets()
	# 		

	# 	if value==3:
	# 		layout.clear_widgets()


		return self.params

				



class DssApp(App):
    def build(self):
        return RootWidget()


if __name__ == '__main__':
    DssApp().run()