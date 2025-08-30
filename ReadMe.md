## 📌 Overview

This project is a sophisticated stock trend prediction application that leverages a Long Short-Term Memory (LSTM) neural network. The core of the application is a trained Keras model (`keras_model.h5`) that analyzes historical stock data to forecast future price movements. The user interface is built with Streamlit, providing an interactive and intuitive way to visualize data and model predictions.

---

## ✨ Features

- **Interactive UI**: A modern and responsive web interface built with Streamlit.
- **Dynamic Stock Data**: Fetches real-time and historical stock data from Yahoo Finance.
- **Data Visualization**: Interactive charts using Plotly to display closing prices, moving averages, and predictions.
- **LSTM Model**: Utilizes a pre-trained `keras_model.h5` for fast and accurate trend prediction.
- **Easy-to-Use**: Simply enter a stock ticker to generate predictions and visualizations.

---

## 📁 Project Structure

.
├── sample.py               # The main Streamlit application script.
├── Untitled.ipynb          # Jupyter Notebook containing model training and experimentation.
├── keras_model.h5          # The pre-trained LSTM model file.
├── requirements.txt        # List of Python dependencies.
└── README.md               # This file.


---

## 🚀 Getting Started

### Prerequisites

- Python 3.1 or higher.
- `pip` package manager.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install dependencies:**
    The project relies on a few key libraries. You can install them all at once using the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    This command will install necessary packages including `tensorflow`, `keras`, `streamlit`, `yfinance`, and `plotly` etc.

### How to Run the Application

1.  **Ensure Model File Exists:** Make sure the `keras_model.h5` file is present in the same directory as `sample.py`.

2.  **Run the Streamlit App:**
    Launch the web application from your terminal:
    ```bash
    streamlit run sample.py
    ```

3.  **Access the App:**
    Your web browser will automatically open a new tab with the application at `http://localhost:8501`.

---

## ⚙️ How It Works

The `sample.py` script orchestrates the entire application flow:

1.  **User Input**: Streamlit's interface takes a stock ticker symbol (e.g., `AAPL`, `GOOGL`) from the user.
2.  **Data Retrieval**: The `yfinance` library fetches historical closing prices for the specified ticker.
3.  **Data Preprocessing**: The data is scaled using `MinMaxScaler` to prepare it for the neural network.
4.  **Model Loading**: The pre-trained `keras_model.h5` is loaded into memory.
5.  **Prediction**: The model makes predictions on a test dataset.
6.  **Visualization**: Predictions are compared against original prices and displayed in interactive Plotly charts.

---

## 👩‍🔬 Model Training Details

For details on the model architecture, data split, and training process, please refer to the `Untitled.ipynb` Jupyter Notebook. This notebook serves as the core documentation for the machine learning pipeline used to create `keras_model.h5`.


