from tkinter import filedialog
from models.loadmodel import ModelLoader
from models.trainer import ModelTrainer
import tkinter as tk
import threading
import os
import datetime


class ModelTrainingGUI:
    """
    A class that provides a graphical user interface (GUI) for training and evaluating time series
    forecasting models using CSV data.

    Attributes:
    root (tk.Tk): The main tkinter window for the GUI.
    config (dict): A dictionary containing the configurable parameters for training and evaluating
    the models.
    csv_training_data_filename (str): The path to the selected CSV file for training the model.
    entries (dict): A dictionary containing the input fields for the model training parameters.
    training_filename_entry (tk.Entry): An input field displaying the selected training data file.
    model_filename_entry (tk.Entry): An input field displaying the selected pre-trained model file
    for evaluation.

    Methods:
    _initialize_gui(): Initializes the graphical user interface with appropriate window dimensions
    and title.
    _create_config_entries(): Creates input fields for the user to configure model training
    parameters.
    _create_buttons(): Creates buttons for model training and evaluation.
    _create_filename_entries(): Creates input fields to display the selected training data file and
    pre-trained model file for evaluation.
    update_training_filename_entry(filename: str): Updates the input field with the selected
    training data file.
    update_model_filename_entry(filename: str): Updates the input field with the selected
    pre-trained model file.
    train_model(): Trains the model using the provided configuration parameters and selected
    training data file.
    on_train_click(): Callback for the "Train Model" button, which starts the model training
    process.
    evaluate_model(model_filename: str): Evaluates a pre-trained model using live data and the
    provided configuration parameters.
    on_load_click(): Callback for the "Load Model" button, which starts the model evaluation
    process.
    run(): Starts the tkinter main loop to run the GUI.

    Example usage:

    if name == 'main':
    gui = ModelTrainingGUI()
    gui.run()

    """
    def __init__(self):
        self.root = tk.Tk()
        self.config = {
            "hidden_size": 256,
            "num_layers": 2,
            "num_heads": 4,
            "output_size": 1,
            "learning_rate": 0.0001,
            "weight_decay": 1e-4,
            "dropout": 0.1,
            "sequence_length": 24,
            "batch_size": 128,
            "num_epochs": 10,
            "save_dir": "save"
        }
        self.csv_training_data_filename = None
        self._initialize_gui()

    def _initialize_gui(self):
        self.root.title("Model Training and Evaluation")

        window_width = 800
        window_height = 600
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x_coordinate = int((screen_width / 2) - (window_width / 2))
        y_coordinate = int((screen_height / 2) - (window_height / 2))
        if not os.path.exists(self.config['save_dir']):
            os.makedirs(self.config['save_dir'])
        self.root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")
        self._create_config_entries()
        self._create_buttons()
        self._create_filename_entries()
        self._create_saved_models_listbox()

    def _create_config_entries(self):
        config_labels_entries = [
            ("Hidden Size:", "256", 0, 0),
            ("Number of Layers:", "2", 0, 1),
            ("Number of Heads:", "4", 1, 0),
            ("Output Size:", "1", 1, 1),
            ("Learning Rate:", "0.0001", 2, 0),
            ("Weight Decay:", "1e-4", 2, 1),
            ("Dropout:", "0.1", 3, 0),
            ("Sequence Length:", "24", 3, 1),
            ("Batch Size:", "128", 4, 0),
            ("Number of Epochs:", "10", 4, 1),
        ]

        self.entries = {}
        for label_text, entry_text, row, column in config_labels_entries:
            label = tk.Label(self.root, text=label_text, font=('Arial', 12), bg='#f0f0f0')
            label.grid(row=row, column=column * 2, padx=10, pady=10, sticky=tk.W)

            entry = tk.Entry(self.root, font=('Arial', 12), width=10)
            entry.insert(0, entry_text)
            entry.grid(row=row, column=column * 2 + 1, padx=10, pady=10, sticky=tk.W)
            self.entries[label_text] = entry

    def _create_saved_models_listbox(self):
        models_list_label = tk.Label(self.root, text="Saved Models:", font=('Arial', 12), bg='#f0f0f0')
        models_list_label.grid(row=8, column=0, padx=10, pady=10, columnspan=2, sticky=tk.W)

        self.models_listbox = tk.Listbox(self.root, font=('Arial', 12), width=80, height=10, bg='#e1e1e1',
                                         relief=tk.SUNKEN, bd=2)
        self.models_listbox.grid(row=9, column=0, padx=10, pady=10, columnspan=4, sticky=tk.W)

        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.models_listbox.yview)
        scrollbar.grid(row=9, column=4, pady=10, sticky=tk.NS)

        self.models_listbox.config(yscrollcommand=scrollbar.set)
        self.models_listbox.bind('<Double-Button-1>', self.on_listbox_click)
        self.update_saved_models_list()

    def update_saved_models_list(self):
        saved_models_directory = "./save"
        model_files = [f for f in os.listdir(saved_models_directory) if f.endswith(".pt")]

        self.models_listbox.delete(0, tk.END)
        for model_file in model_files:
            model_path = os.path.join(saved_models_directory, model_file)
            model_ctime = os.path.getctime(model_path)
            model_date = datetime.datetime.fromtimestamp(model_ctime).strftime("%Y-%m-%d %H:%M:%S")
            model_name_with_date = f"{model_file} ({model_date})"
            self.models_listbox.insert(tk.END, model_name_with_date)

    def on_listbox_click(self, event):
        selected_model = self.models_listbox.get(self.models_listbox.curselection())
        model_filename = os.path.join(self.config['save_dir'], selected_model)
        self.update_model_filename_entry(model_filename)
        evaluate_thread = threading.Thread(target=self.evaluate_model, args=(model_filename,))
        evaluate_thread.start()

    def _create_buttons(self):
        train_button = tk.Button(self.root, text="Train Model", command=self.on_train_click, bg='#3db5e6', fg='white',
                                 font=('Arial', 14), relief=tk.GROOVE, bd=2)
        train_button.grid(row=5, column=0, padx=10, pady=10, columnspan=2)

        load_button = tk.Button(self.root, text="Load Model", command=self.on_load_click, bg='#3db5e6', fg='white',
                                font=('Arial', 14), relief=tk.GROOVE, bd=2)
        load_button.grid(row=5, column=2, padx=10, pady=10, columnspan=2)

    def _create_filename_entries(self):
        training_data_label = tk.Label(self.root, text=f"CSV file for training:", font=('Arial', 12), bg='#f0f0f0')
        training_data_label.grid(row=6, column=0, padx=10, pady=10, columnspan=2, sticky=tk.W)

        self.training_filename_entry = tk.Entry(self.root, font=('Arial', 12), width=80)
        self.training_filename_entry.insert(0, "Click 'Train Model' button to select a file")
        self.training_filename_entry.grid(row=6, column=2, padx=10, pady=10, columnspan=2)

        model_filename_label = tk.Label(self.root, text="Trained model to load:", font=('Arial', 12), bg='#f0f0f0')
        model_filename_label.grid(row=7, column=0, padx=10, pady=10, columnspan=2, sticky=tk.W)

        self.model_filename_entry = tk.Entry(self.root, font=('Arial', 12), width=80)
        self.model_filename_entry.insert(0, "Click 'Load Model' button to select a file")
        self.model_filename_entry.grid(row=7, column=2, padx=10, pady=10, columnspan=2)
        self.training_filename_entry.bind('<Return>', self.on_training_filename_entry_key)
        self.model_filename_entry.bind('<Return>', self.on_model_filename_entry_key)

    def update_training_filename_entry(self, filename):
        self.training_filename_entry.delete(0, tk.END)
        self.training_filename_entry.insert(0, filename)

    def update_model_filename_entry(self, filename):
        self.model_filename_entry.delete(0, tk.END)
        self.model_filename_entry.insert(0, filename)

    def on_training_filename_entry_key(self, event):
        self.csv_training_data_filename = self.training_filename_entry.get()
        if self.csv_training_data_filename:
            train_thread = threading.Thread(target=self.train_model)
            train_thread.start()

    def on_model_filename_entry_key(self, event):
        model_filename = self.model_filename_entry.get()
        if model_filename:
            evaluate_thread = threading.Thread(target=self.evaluate_model, args=(model_filename,))
            evaluate_thread.start()

    def train_model(self):
        try:
            self.config['hidden_size'] = int(self.entries["Hidden Size:"].get())
            self.config['num_layers'] = int(self.entries["Number of Layers:"].get())
            self.config['num_heads'] = int(self.entries["Number of Heads:"].get())
            self.config['output_size'] = int(self.entries["Output Size:"].get())
            self.config['learning_rate'] = float(self.entries["Learning Rate:"].get())
            self.config['weight_decay'] = float(self.entries["Weight Decay:"].get())
            self.config['dropout'] = float(self.entries["Dropout:"].get())
            self.config['sequence_length'] = int(self.entries["Sequence Length:"].get())
            self.config['batch_size'] = int(self.entries["Batch Size:"].get())
            self.config['num_epochs'] = int(self.entries["Number of Epochs:"].get())

            trainer = ModelTrainer(self.config)
            trainer.preprocess_data(self.csv_training_data_filename, chunksize=10000, input_type='file')
            trainer.split_data(test_size=0.1, random_state=42)
            trainer.configure_model()
            trainer.train_model()
            trainer.save_model()
            trainer.test_model()
            trainer.evaluate_model(tail_n=200)
        except Exception as e:
            self.root.after(0, print, f"Error while training the model: {e}")

    def on_train_click(self):
        self.csv_training_data_filename = filedialog.askopenfilename(initialdir="./",
                                                                     title="Select CSV file for training")
        if self.csv_training_data_filename:
            self.update_training_filename_entry(self.csv_training_data_filename)
            train_thread = threading.Thread(target=self.train_model)
            train_thread.start()

    def evaluate_model(self, model_filename):
        try:
            self.config['hidden_size'] = int(self.entries["Hidden Size:"].get())
            self.config['num_layers'] = int(self.entries["Number of Layers:"].get())
            self.config['num_heads'] = int(self.entries["Number of Heads:"].get())
            self.config['output_size'] = int(self.entries["Output Size:"].get())
            self.config['learning_rate'] = float(self.entries["Learning Rate:"].get())
            self.config['weight_decay'] = float(self.entries["Weight Decay:"].get())
            self.config['dropout'] = float(self.entries["Dropout:"].get())
            self.config['sequence_length'] = int(self.entries["Sequence Length:"].get())
            self.config['batch_size'] = int(self.entries["Batch Size:"].get())
            self.config['num_epochs'] = int(self.entries["Number of Epochs:"].get())

            model_loader = ModelLoader(self.config)
            data, input_size = model_loader.fetch_live_data()
            self.config['input_size'] = input_size
            model_loader.preprocess_data(data, chunksize=1000, input_type='dataframe')
            model_loader.configure_model()
            model_loader.load_and_evaluate(model_filename, tail_n=200)
        except Exception as e:
            self.root.after(0, print, f"Error while evaluating the model: {e}")

    def on_load_click(self):
        model_filename = filedialog.askopenfilename(initialdir="./", title="Select trained model to load")
        if model_filename:
            self.update_model_filename_entry(model_filename)
            evaluate_thread = threading.Thread(target=self.evaluate_model, args=(model_filename,))
            evaluate_thread.start()

    def run(self):
        self.root.mainloop()

