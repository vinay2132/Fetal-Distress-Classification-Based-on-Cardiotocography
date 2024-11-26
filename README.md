# Fetal Distress Classification Using Cardiotocography

## Overview

This project provides a machine learning-based solution for predicting fetal distress conditions (Normal, Suspicious, or Pathological) using Cardiotocography (CTG) data. The goal is to assist medical professionals in analyzing fetal health efficiently and accurately, ensuring timely interventions during pregnancy and childbirth.

Cardiotocography monitors fetal heart rate and uterine contractions to detect distress signals. Using Random Forest and other machine learning algorithms, this project achieves high accuracy in classifying fetal health states.

---

## Features

- **Data Preprocessing**: Handles missing values, removes irrelevant features, and standardizes data using feature scaling.
- **Visualization**: Provides insights through correlation maps and data distribution plots.
- **Classification**: Implements Random Forest for classification with an approximate accuracy of 95%.
- **Web Interface**: A user-friendly HTML and Flask-based UI for inputting data and viewing results.
- **Integration**: Supports integration with clinical workflows through a robust backend built with Python and Flask.

---

## Technologies Used

- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Flask, Matplotlib, Seaborn
- **Machine Learning Algorithm**: Random Forest
- **Web Framework**: Flask
- **Visualization Tools**: Matplotlib, Seaborn

---

## Dataset

The CTG dataset used for training and evaluation is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Cardiotocography). It contains 2,126 observations with 21 features, categorized into three classes:

- Normal: 1,655 instances
- Suspicious: 295 instances
- Pathological: 176 instances

---

## How It Works

1. **Data Input**: The user provides inputs from the CTG machine (e.g., baseline value, accelerations, uterine contractions).
2. **Processing**: The data is standardized and passed through the trained Random Forest model.
3. **Prediction**: The system predicts the fetal condition as Normal, Suspicious, or Pathological.
4. **Output**: The prediction, along with additional insights and recommended actions, is displayed on the UI.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fetal-distress-classification.git
   cd fetal-distress-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Access the application:
   Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## Project Structure

```
├── app.py               # Flask application
├── model.py             # Model training and evaluation script
├── templates/           # HTML templates for the web interface
├── static/              # CSS and other static assets
├── fetal_health.csv     # CTG dataset
├── modelrf.pkl          # Pre-trained Random Forest model
├── scaler.pkl           # Pre-trained scaler for feature normalization
├── README.md            # Project documentation
└── requirements.txt     # List of dependencies
```

---

## Results

- **Accuracy**: 95%
- **Precision**: 93%
- **Recall**: 91%
- **F1 Score**: Balanced at 92%

The Random Forest model outperformed decision tree and k-nearest neighbors approaches previously evaluated for this dataset.

---

## Contributors

- Poojitha Cherukuri
- Vinay Daram
- Venkata Surya Mouli Sree Vamsi Mallady
- Aravind Reddy Pallreddy
- Madhu Surisetti

---

## Future Enhancements

- Integration with cloud-based platforms for scalability.
- Support for additional machine learning models.
- Improved UI for real-time data entry and analysis.
- Addition of new datasets to enhance model generalizability.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
