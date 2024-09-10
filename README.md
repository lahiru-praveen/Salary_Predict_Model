# Salary Prediction Model

This is a test project that demonstrates a simple linear regression model to predict salaries based on years of experience. The dataset used in this project was sourced from Kaggle. The project includes a Streamlit web application that allows users to input their years of experience and get the predicted salary. Additionally, the app provides visualizations and model performance metrics.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Acknowledgements](#acknowledgements)

## Overview

This project aims to predict salaries using a simple linear regression model. The primary objective is to understand the relationship between years of experience and salary, and to create a user-friendly web application where users can interact with the model.

## Dataset

The dataset used in this project is sourced from Kaggle and contains the following columns:

- `YearsExperience`: Number of years of professional experience.
- `Salary`: Corresponding annual salary.

You can find the original dataset on Kaggle [here](https://www.kaggle.com).

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/salary-predictor.git
    cd salary-predictor
    ```

2. **Create a virtual environment and activate it:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app:**

    ```bash
    streamlit run model.py
    ```

## Usage

Once the app is running, open your web browser and go to `http://localhost:8501`. You can input the number of years of experience in the sidebar and see the predicted salary in real-time. The app also provides visualizations and details about the model's performance.

## Project Structure

```plaintext
salary-predictor/
│
├── Salary Data.csv       # Dataset file
├── model.py                # Main Streamlit app file
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── .gitignore            # Git ignore file
