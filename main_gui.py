import tkinter as tk
from tkinter import filedialog
import threading
from models.load_model import ModelLoader
from models.trainer_model import ModelTrainer


def start_gui():
    config = {
        "hidden_size": 256,
        "num_layers": 2,
        "num_heads": 4,
        "output_size": 1,
        "learning_rate": 0.0001,
        "weight_decay": 1e-4,
        "dropout": 0.1,
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
            config['hidden_size'] = int(hidden_size_entry.get())
            config['num_layers'] = int(num_layers_entry.get())
            config['num_heads'] = int(num_heads_entry.get())
            config['output_size'] = int(output_size_entry.get())
            config['learning_rate'] = float(learning_rate_entry.get())
            config['weight_decay'] = float(weight_decay_entry.get())
            config['dropout'] = float(dropout_entry.get())
            config['sequence_length'] = int(sequence_length_entry.get())
            config['batch_size'] = int(batch_size_entry.get())
            config['num_epochs'] = int(num_epochs_entry.get())
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
            config['hidden_size'] = int(hidden_size_entry.get())
            config['num_layers'] = int(num_layers_entry.get())
            config['num_heads'] = int(num_heads_entry.get())
            config['output_size'] = int(output_size_entry.get())
            config['learning_rate'] = float(learning_rate_entry.get())
            config['weight_decay'] = float(weight_decay_entry.get())
            config['dropout'] = float(dropout_entry.get())
            config['sequence_length'] = int(sequence_length_entry.get())
            config['batch_size'] = int(batch_size_entry.get())
            config['num_epochs'] = int(num_epochs_entry.get())
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
    # Use grid layout instead of pack layout
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

    for label_text, entry_text, row, column in config_labels_entries:
        label = tk.Label(root, text=label_text, font=('Arial', 12))
        label.grid(row=row, column=column * 2, padx=10, pady=10, sticky=tk.W)

        entry = tk.Entry(root, font=('Arial', 12), width=10)
        entry.insert(0, entry_text)
        entry.grid(row=row, column=column * 2 + 1, padx=10, pady=10, sticky=tk.W)

        if label_text == "Hidden Size:":
            hidden_size_entry = entry
        elif label_text == "Number of Layers:":
            num_layers_entry = entry
        elif label_text == "Number of Heads:":
            num_heads_entry = entry
        elif label_text == "Output Size:":
            output_size_entry = entry
        elif label_text == "Learning Rate:":
            learning_rate_entry = entry
        elif label_text == "Weight Decay:":
            weight_decay_entry = entry
        elif label_text == "Dropout:":
            dropout_entry = entry
        elif label_text == "Sequence Length:":
            sequence_length_entry = entry
        elif label_text == "Batch Size:":
            batch_size_entry = entry
        elif label_text == "Number of Epochs:":
            num_epochs_entry = entry

    train_button = tk.Button(root, text="Train Model", command=on_train_click, bg='#3db5e6', fg='white',
                             font=('Arial', 14))
    train_button.grid(row=5, column=0, padx=10, pady=10, columnspan=2)

    load_button = tk.Button(root, text="Load Model", command=on_load_click, bg='#3db5e6', fg='white', font=('Arial', 14))
    load_button.grid(row=5, column=2, padx=10, pady=10, columnspan=2)

    training_data_label = tk.Label(root, text=f"CSV file for training:", font=('Arial', 12))
    training_data_label.grid(row=6, column=0, padx=10, pady=10, columnspan=2, sticky=tk.W)
    training_filename_entry = tk.Entry(root, font=('Arial', 12), width=80)
    training_filename_entry.insert(0, "Click 'Train Model' button to select a file")
    training_filename_entry.grid(row=6, column=2, padx=10, pady=10, columnspan=2)

    model_filename_label = tk.Label(root, text="Trained model to load:", font=('Arial', 12))
    model_filename_label.grid(row=7, column=0, padx=10, pady=10, columnspan=2, sticky=tk.W)

    model_filename_entry = tk.Entry(root, font=('Arial', 12), width=80)
    model_filename_entry.insert(0, "Click 'Load Model' button to select a file")
    model_filename_entry.grid(row=7, column=2, padx=10, pady=10, columnspan=2)

    root.mainloop()


if __name__ == '__main__':
    start_gui()
