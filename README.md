
# Cryptocurrency Dashboard

This is a Dash application for visualizing cryptocurrency data. Follow the instructions below to set up and run the application.

## Table of Contents
- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Notes](#notes)
- [Team](#team)
- [Instructor](#instructor)
- [Module](#module)

## Project Overview
The Cryptocurrency Dashboard project is an AI-driven application designed to analyze and visualize trading data for various cryptocurrencies. It incorporates advanced machine learning models and technical indicators to provide users with insights and predictions.

Demo video here: [https://drive.google.com/drive/folders/1XpqXTdwXp8Vbwwzaqk7UhYqVg2_CVcLY?usp=drive_link](https://drive.google.com/drive/folders/1XpqXTdwXp8Vbwwzaqk7UhYqVg2_CVcLY?usp=drive_link)

## Tech Stack
- Plotly Dash, Bootstrap
- Python, Pandas, Numpy, Keras, Tensorflow, Websocket

## Features
- **Visualize Cryptocurrency Data**: Provides interactive visualizations of cryptocurrency trading data.
- **Technical Indicators**: Displays various technical indicators such as Support and Resistance, Rate of Change (ROC), Moving Average (MA), Relative Strength Index (RSI), Bollinger Bands (BB), and Moving Average Convergence/Divergence (MACD).
- **Machine Learning Models**: Utilizes models like RNN, LSTM, XGBoost, Transformer and Time Embeddings, and CNN for predictive analysis.
- **Real-time Data**: Integrates real-time data using Websocket.

## Prerequisites
Make sure you have Python installed on your computer. If not, download and install Python from [python.org](https://www.python.org/).

## Setup Instructions

### 1. Clone the Repository

Clone this repository to your local machine using:
```bash
git clone https://github.com/luuquang200/CryptoProject.git
cd CryptoProject
```

### 2. Create a Virtual Environment

Open a terminal (or Command Prompt on Windows) and navigate to the project directory. Create a virtual environment using the following command:

```bash
python -m venv venv
```
`venv` is the name of the virtual environment. You can choose a different name if you prefer.

### 3. Activate the Virtual Environment

Activate the virtual environment using the following command:

- On Windows:
  ```bash
  .\venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### 4. Install Required Packages

Make sure you have a `requirements.txt` file that lists all the necessary packages and their versions. Install the required packages using:

```bash
pip install -r requirements.txt
```

### 5. Run the Application

After activating the virtual environment and installing the packages, run the application using:

```bash
python app_dev.py
```

### 6. Deactivate the Virtual Environment

Once you are done, you can deactivate the virtual environment using:

```bash
deactivate
```

## Usage
To use the Cryptocurrency Dashboard application, follow these steps:

1. **Access the application**: Open your web browser and go to `http://localhost:8050` to access the application locally.
2. **Explore features**: Use the interactive features to visualize cryptocurrency data, analyze technical indicators, and view predictions from machine learning models.
3. **Update data**: Ensure your data sources are connected for real-time updates via Websocket.

## Notes
- Ensure you have created a `.env` file with necessary environment variables if the application depends on it.
- If you encounter any issues, make sure all the dependencies are correctly listed in the `requirements.txt` file.

## Team
- **Student ID: 20120454** - Cong-Dat Le
- **Student ID: 20120489** - Phi-Hung Vo
- **Student ID: 20120558** - Ngoc-Quang Luu
- **Student ID: 20120582** - Huu-Thanh Tran

## Instructor
- M.S. Van-Quy Tran
- M.S. Duy-Quang Tran
- M.S. Nguyen-Kha Do

## Module
**Advanced Topics in Software Development Technology**
