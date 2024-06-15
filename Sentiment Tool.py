import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

class SentimentAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analysis Models")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        self.setup_ui()

    def setup_ui(self):
        self.root.configure(bg='#f0f0f0')

        self.title_label = tk.Label(self.root, text="Sentiment Analysis Models", font=("Helvetica", 16, "bold"), bg='#f0f0f0')
        self.title_label.pack(pady=10)

        self.load_button = tk.Button(self.root, text="Load Dataset", command=self.load_dataset, bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"))
        self.load_button.pack(pady=10)

        self.progress = Progressbar(self.root, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(pady=10)

        self.model_buttons_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.model_buttons_frame.pack(pady=10)

        self.lr_button = tk.Button(self.model_buttons_frame, text="Logistic Regression", command=self.run_lr, bg="#2196F3", fg="white", font=("Helvetica", 12, "bold"))
        self.lr_button.grid(row=0, column=0, padx=5)

        self.svm_button = tk.Button(self.model_buttons_frame, text="SVM", command=self.run_svm, bg="#FF5722", fg="white", font=("Helvetica", 12, "bold"))
        self.svm_button.grid(row=0, column=1, padx=5)

        self.lstm_button = tk.Button(self.model_buttons_frame, text="LSTM", command=self.run_lstm, bg="#9C27B0", fg="white", font=("Helvetica", 12, "bold"))
        self.lstm_button.grid(row=0, column=2, padx=5)

        self.graphs_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.graphs_frame.pack(pady=10)

        self.lr_graph_button = tk.Button(self.graphs_frame, text="Show LR Graphs", command=self.show_lr_graphs, state=tk.DISABLED, bg="#795548", fg="white", font=("Helvetica", 12, "bold"))
        self.lr_graph_button.grid(row=0, column=0, padx=5)

        self.svm_graph_button = tk.Button(self.graphs_frame, text="Show SVM Graphs", command=self.show_svm_graphs, state=tk.DISABLED, bg="#607D8B", fg="white", font=("Helvetica", 12, "bold"))
        self.svm_graph_button.grid(row=0, column=1, padx=5)

        self.lstm_graph_button = tk.Button(self.graphs_frame, text="Show LSTM Graphs", command=self.show_lstm_graphs, state=tk.DISABLED, bg="#009688", fg="white", font=("Helvetica", 12, "bold"))
        self.lstm_graph_button.grid(row=0, column=2, padx=5)

        self.summary_button = tk.Button(self.root, text="Show Model Comparison", command=self.show_comparison, state=tk.DISABLED, bg="#FF9800", fg="white", font=("Helvetica", 12, "bold"))
        self.summary_button.pack(pady=10)

        self.predict_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.predict_frame.pack(pady=10)
        
        self.predict_label = tk.Label(self.predict_frame, text="Enter text to predict sentiment:", bg='#f0f0f0', font=("Helvetica", 12))
        self.predict_label.grid(row=0, column=0, padx=5)
        
        self.predict_entry = tk.Entry(self.predict_frame, width=50, font=("Helvetica", 12))
        self.predict_entry.grid(row=0, column=1, padx=5)
        
        self.predict_button = tk.Button(self.predict_frame, text="Predict", command=self.predict_sentiment, bg="#3F51B5", fg="white", font=("Helvetica", 12, "bold"))
        self.predict_button.grid(row=0, column=2, padx=5)

        self.result_label = tk.Label(self.root, text="", bg='#f0f0f0', font=("Helvetica", 12))
        self.result_label.pack(pady=10)

    def load_dataset(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        
        self.data = pd.read_csv(file_path)
        self.data['clean_text'] = self.data['text'].apply(self.clean_text)
        
        label_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
        self.data['sentiment'] = self.data['sentiment'].map(label_mapping)

        self.X = self.data['clean_text']
        self.y = self.data['sentiment']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        self.X_test_tfidf = self.vectorizer.transform(self.X_test)

        self.tokenizer = Tokenizer(num_words=5000)
        self.tokenizer.fit_on_texts(self.X_train)
        self.X_train_seq = self.tokenizer.texts_to_sequences(self.X_train)
        self.X_test_seq = self.tokenizer.texts_to_sequences(self.X_test)
        self.X_train_pad = pad_sequences(self.X_train_seq, maxlen=100)
        self.X_test_pad = pad_sequences(self.X_test_seq, maxlen=100)

        messagebox.showinfo("Info", "Dataset loaded and preprocessed successfully!")

    def clean_text(self, text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
        text = text.lower()
        return text

    def run_lr(self):
        self.update_progress(20)
        self.lr_model = LogisticRegression()
        self.lr_model.fit(self.X_train_tfidf, self.y_train)
        self.lr_preds = self.lr_model.predict(self.X_test_tfidf)
        self.lr_accuracy = accuracy_score(self.y_test, self.lr_preds)
        self.update_progress(40)
        messagebox.showinfo("Logistic Regression", f"Accuracy: {self.lr_accuracy:.2f}\n\n{classification_report(self.y_test, self.lr_preds)}")
        self.lr_graph_button.config(state=tk.NORMAL)
        self.update_progress(100)
        self.summary_button.config(state=tk.NORMAL)

    def run_svm(self):
        self.update_progress(20)
        self.svm_model = SVC()
        self.svm_model.fit(self.X_train_tfidf, self.y_train)
        self.svm_preds = self.svm_model.predict(self.X_test_tfidf)
        self.svm_accuracy = accuracy_score(self.y_test, self.svm_preds)
        self.update_progress(40)
        messagebox.showinfo("SVM", f"Accuracy: {self.svm_accuracy:.2f}\n\n{classification_report(self.y_test, self.svm_preds)}")
        self.svm_graph_button.config(state=tk.NORMAL)
        self.update_progress(100)
        self.summary_button.config(state=tk.NORMAL)

    def run_lstm(self):
        self.update_progress(20)
        self.lstm_model = Sequential()
        self.lstm_model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
        self.lstm_model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        self.lstm_model.add(Dense(3, activation='softmax'))
        self.lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.lstm_model.fit(self.X_train_pad, self.y_train, epochs=5, batch_size=64, validation_data=(self.X_test_pad, self.y_test))
        
        self.lstm_preds_prob = self.lstm_model.predict(self.X_test_pad)
        self.lstm_preds = np.argmax(self.lstm_preds_prob, axis=1)
        self.lstm_accuracy = accuracy_score(self.y_test, self.lstm_preds)
        self.update_progress(40)
        messagebox.showinfo("LSTM", f"Accuracy: {self.lstm_accuracy:.2f}\n\n{classification_report(self.y_test, self.lstm_preds)}")
        self.lstm_graph_button.config(state=tk.NORMAL)
        self.update_progress(100)
        self.summary_button.config(state=tk.NORMAL)

    def update_progress(self, value):
        self.progress['value'] = value
        self.root.update_idletasks()

    def show_lr_graphs(self):
        self.plot_confusion_matrix(self.y_test, self.lr_preds, 'Confusion Matrix for Logistic Regression')

    def show_svm_graphs(self):
        self.plot_confusion_matrix(self.y_test, self.svm_preds, 'Confusion Matrix for SVM')

    def show_lstm_graphs(self):
        self.plot_confusion_matrix(self.y_test, self.lstm_preds, 'Confusion Matrix for LSTM')

    def plot_confusion_matrix(self, y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(title)
        plt.show()

    def show_comparison(self):
        accuracies = [self.lr_accuracy, self.svm_accuracy, self.lstm_accuracy]
        models = ['Logistic Regression', 'SVM', 'LSTM']

        plt.figure(figsize=(10, 6))
        plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of Different Models')
        plt.ylim(0, 1)
        plt.show()

    def predict_sentiment(self):
        text = self.predict_entry.get()
        if not text:
            messagebox.showwarning("Input Error", "Please enter some text to predict sentiment.")
            return
        
        clean_text = self.clean_text(text)
        text_tfidf = self.vectorizer.transform([clean_text])
        text_seq = self.tokenizer.texts_to_sequences([clean_text])
        text_pad = pad_sequences(text_seq, maxlen=100)
        
        lr_pred = self.lr_model.predict(text_tfidf)[0]
        svm_pred = self.svm_model.predict(text_tfidf)[0]
        lstm_pred = np.argmax(self.lstm_model.predict(text_pad), axis=1)[0]
        
        sentiments = ['Negative', 'Neutral', 'Positive']
        result = f"Logistic Regression: {sentiments[lr_pred]}\nSVM: {sentiments[svm_pred]}\nLSTM: {sentiments[lstm_pred]}"
        self.result_label.config(text=result)

if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentAnalysisApp(root)
    root.mainloop()
