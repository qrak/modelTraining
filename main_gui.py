import tkinter as tk
from tkinter import filedialog
import threading
import torch
from models.load_model import ModelLoader
from models.trainer_model import ModelTrainer


def start_gui():
    config = {
        "hidden_size": 256,
        "num_layers": 2,
        "num_heads": 2,
        "output_size": 1,
        "learning_rate": 0.0001,
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "sequence_length": 24,
        "batch_size": 128,
        "num_epochs": 10
    }

    def update_training_filename_entry(filename):
        training_filename_entry.delete(0, tk.END)
        training_filename_entry.insert(0, filename)

    def update_model_filename_entry(filename):
        model_filename_entry.delete(0, tk.END)
        model_filename_entry.insert(0, filename)

    def train_model():
        try:
            trainer = ModelTrainer(config)
            trainer.preprocess_data(csv_training_data_filename, chunksize=10000, input_type='file')
            trainer.split_data(test_size=0.1, random_state=42)
            trainer.configure_model()
            trainer.train_model()
            trainer.save_model()
            trainer.test_model()
            trainer.evaluate_model(tail_n=200)
        except Exception as e:
            root.after(0, print, f"Error while training the model: {e}")

    def on_train_click():
        global csv_training_data_filename
        csv_training_data_filename = filedialog.askopenfilename(initialdir="./", title="Select CSV file for training")
        if csv_training_data_filename:
            update_training_filename_entry(csv_training_data_filename)
            train_thread = threading.Thread(target=train_model)
            train_thread.start()

    def evaluate_model(model_filename):
        try:
            model_loader = ModelLoader(config)
            data, input_size = model_loader.fetch_live_data()
            config['input_size'] = input_size
            model_loader.preprocess_data(data, chunksize=1000, input_type='dataframe')
            model_loader.configure_model()
            model_loader.load_and_evaluate(model_filename, tail_n=200)
        except Exception as e:
            root.after(0, print, f"Error while evaluating the model: {e}")

    def on_load_click():
        model_filename = filedialog.askopenfilename(initialdir="./", title="Select trained model to load")
        if model_filename:
            update_model_filename_entry(model_filename)
            evaluate_thread = threading.Thread(target=evaluate_model, args=(model_filename,))
            evaluate_thread.start()

    root = tk.Tk()
    root.title("Model Training and Evaluation")

    window_width = 800
    window_height = 600
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x_coordinate = int((screen_width / 2) - (window_width / 2))
    y_coordinate = int((screen_height / 2) - (window_height / 2))
    root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

    train_button = tk.Button(root, text="Train Model", command=on_train_click, bg='#3db5e6', fg='white',
                             font=('Arial', 14))
    train_button.pack(padx=10, pady=10)

    load_button = tk.Button(root, text="Load Model", command=on_load_click, bg='#3db5e6', fg='white', font=('Arial', 14))
    load_button.pack(padx=10, pady=10)

    training_data_label = tk.Label(root, text=f"CSV file for training:", font=('Arial', 12))
    training_data_label.pack(padx=10, pady=10)
    training_filename_entry = tk.Entry(root, font=('Arial', 12), width=80)
    training_filename_entry.insert(0, "Click 'Train Model' button to select a file")
    training_filename_entry.pack(padx=10, pady=10)

    model_filename_label = tk.Label(root, text="Trained model to load:", font=('Arial', 12))
    model_filename_label.pack(padx=10, pady=10)

    model_filename_entry = tk.Entry(root, font=('Arial', 12), width=80)
    model_filename_entry.insert(0, "Click 'Load Model' button to select a file")
    model_filename_entry.pack(padx=10, pady=10)

    root.mainloop()


if __name__ == '__main__':
    start_gui()
