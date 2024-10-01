import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import datetime
import numpy as np

class ModelEnsemble:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.y_train = None
        self.df = None

    def train(self, X_train, y_train):
        # Initialize individual models
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        ada_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50)
        gb_model = GradientBoostingClassifier(n_estimators=100)

        # Create a voting classifier with the individual models
        self.model = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('ada', ada_model),
                ('gb', gb_model)
            ],
            voting='soft'
        )

        # Train the voting classifier
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X_test)

class SalesDataApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sales Data Ensemble Learning")
        self.root.geometry("800x600")  # Increase window size

        # Create the layout
        self.frame = ttk.Frame(root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Load button
        self.load_button = ttk.Button(self.frame, text="Load CSV", command=self.load_csv, style='Load.TButton')
        self.load_button.grid(row=0, column=0, pady=10)

        # Predict button
        self.predict_button = ttk.Button(self.frame, text="Predict Sales", command=self.predict_sales, style='Predict.TButton')
        self.predict_button.grid(row=0, column=1, pady=10)
        self.predict_button.config(state=tk.DISABLED)

        # Result label
        self.result_label = ttk.Label(self.frame, text="Accuracy: N/A", font=("Arial", 12))
        self.result_label.grid(row=1, column=0, columnspan=2, pady=10)

        # Date selection
        self.date_label = ttk.Label(self.frame, text="Select Date (YYYY-MM-DD):", font=("Arial", 12))
        self.date_label.grid(row=2, column=0, pady=10)
        self.date_entry = ttk.Entry(self.frame)
        self.date_entry.grid(row=2, column=1, pady=10)

        # Add date picker
        self.date_picker_button = ttk.Button(self.frame, text="Select Date", command=self.select_date, style='DatePicker.TButton')
        self.date_picker_button.grid(row=2, column=2, pady=10)

        # Initialize ensemble model
        self.ensemble_model = ModelEnsemble()
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None

        # Configure styles
        self.style = ttk.Style()
        self.style.configure('Load.TButton', background='lightblue')
        self.style.configure('Predict.TButton', background='lightgreen')
        self.style.configure('DatePicker.TButton', background='lightcoral')

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            # Ensure 'Date' column is in datetime format
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
            # Preprocess data
            # Convert categorical columns to numerical (One-hot encoding)
            self.df = pd.get_dummies(self.df, columns=['Product', 'Region'])
            
            # Define features and target
            X = self.df.drop(['Date', 'Target'], axis=1)
            y = self.df['Target']
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the ensemble model
            self.ensemble_model.train(self.X_train, self.y_train)

            # Enable the predict button
            self.predict_button.config(state=tk.NORMAL)

            messagebox.showinfo("Success", "CSV file loaded and model trained successfully!")

    def predict_sales(self):
        if self.X_test is not None:
            y_pred = self.ensemble_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            self.result_label.config(text=f"Accuracy: {accuracy:.2f}")

            # Plot sales data
            self.plot_sales_data()
        else:
            messagebox.showerror("Error", "No data to predict. Please load a CSV file first.")

    def select_date(self):
        date_str = self.date_entry.get()
        try:
            date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            # Filter data based on selected date
            filtered_data = self.df[self.df['Date'] == date]
            if not filtered_data.empty:
                self.show_filtered_data(filtered_data)
            else:
                messagebox.showwarning("No Data", "No data found for the selected date.")
        except ValueError:
            messagebox.showerror("Invalid Date", "Please enter a valid date in YYYY-MM-DD format.")

    def show_filtered_data(self, data):
        fig, ax = plt.subplots(figsize=(8, 4))
        data.groupby('Product').agg({'Sales': 'sum'}).plot(kind='bar', ax=ax, color=['blue', 'green', 'red'])
        ax.set_title('Sales by Product')
        ax.set_xlabel('Product')
        ax.set_ylabel('Total Sales')
        
        # Display the plot in Tkinter window
        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().grid(row=3, column=0, columnspan=3, pady=20)

    def plot_sales_data(self):
        # Plot overall sales data
        fig, ax = plt.subplots(figsize=(8, 4))
        self.df.groupby('Date').agg({'Sales': 'sum'}).plot(kind='line', ax=ax, color='blue')
        ax.set_title('Sales Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Total Sales')
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        # Display the plot in Tkinter window
        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().grid(row=3, column=0, columnspan=3, pady=20)

# Initialize the Tkinter window
root = tk.Tk()
app = SalesDataApp(root)

# Start the Tkinter event loop
root.mainloop()



