# Bank ML - Machine Learning Course Project

A comprehensive machine learning project focusing on banking data analysis, customer segmentation, fraud detection, and product recommendations.

## Project Structure

```
Bank_ML/
├── data/                          # Raw and processed data
│   └── bank_transactions_data_2.csv
├── notebooks/                     # Jupyter notebooks for exploration and analysis
├── src/                           # Source code modules
│   ├── data_preprocessing.py      # Data loading and cleaning
│   ├── feature_engineering.py     # Feature creation and transformation
│   ├── clustering.py              # Customer segmentation
│   ├── fraud_detection.py         # Fraud classification models
│   └── recommendation.py          # Product recommendation engine
├── app/                           # Application code
│   └── streamlit_app.py          # Interactive web dashboard
├── outputs/                       # Model results and artifacts
├── requirements.txt               # Project dependencies
└── README.md                      # This file
```

## Modules Overview

### 1. Data Preprocessing (`src/data_preprocessing.py`)
Handles data loading, cleaning, validation, and preparation for analysis.

### 2. Feature Engineering (`src/feature_engineering.py`)
Creates and transforms features for machine learning models.

### 3. Clustering (`src/clustering.py`)
Implements customer segmentation using unsupervised learning techniques.

### 4. Fraud Detection (`src/fraud_detection.py`)
Develops classification models to identify fraudulent transactions.

### 5. Recommendation Engine (`src/recommendation.py`)
Builds systems to recommend products and services to customers.

### 6. Streamlit App (`app/streamlit_app.py`)
Provides an interactive web-based interface for model exploration and predictions.

## Getting Started

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. Clone or download this project
2. Navigate to the project directory
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
```bash
streamlit run app/streamlit_app.py
```

## Development Notes

This is a structured foundation for a machine learning course project. Each module is designed to be developed incrementally:

1. Start with **data_preprocessing.py** to understand and clean the data
2. Move to **feature_engineering.py** to create meaningful features
3. Implement **clustering.py** for customer segmentation
4. Develop **fraud_detection.py** for transaction classification
5. Build **recommendation.py** for business recommendations
6. Integrate all modules into **streamlit_app.py** for visualization

## Project Goals

- Understand banking transaction patterns
- Segment customers based on behavior
- Detect and prevent fraudulent transactions
- Provide personalized product recommendations
- Create an interactive analytics dashboard

## Author
ML Course Project

## License
Educational Use
