# Cryptocurrency Dashboard

This is a Dash application for visualizing cryptocurrency data. Follow the instructions below to set up and run the application.

## Prerequisites

Make sure you have Python installed on your computer. If not, download and install Python from [python.org](https://www.python.org/).

## Setup Instructions

### 1. Clone the Repository

Clone this repository to your local machine using:
```bash
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

## Notes

- Ensure you have created a `.env` file with necessary environment variables if the application depends on it.
- If you encounter any issues, make sure all the dependencies are correctly listed in the `requirements.txt` file.

---
