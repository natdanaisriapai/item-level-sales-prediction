# Item-Level Sales Prediction

This project implements a machine learning solution for predicting item-level sales across multiple stores. It uses historical sales data to forecast future sales for each store-item combination.

## Setup

1. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```
item-level sales prediction/
├── data/
│   ├── test.csv        # Test dataset for predictions
│   └── train.csv       # Training dataset with historical sales
├── utils.py            # Utility functions for model training
├── main.ipynb          # Main notebook containing analysis and model
├── model_checkpoints/  # Saved model checkpoints from batch training
│   ├── LightGBM_batch_*.pkl
│   ├── RandomForest_batch_*.pkl
│   └── GradientBoosting_batch_*.pkl
└── requirements.txt    # Python package dependencies
```

## Data Description

The dataset includes:
* Daily sales data for multiple stores and items
* Features: date, store, item, sales
* Training period: 2013-2017
* Prediction target: 3 months ahead (2018 Q1)

## Methodology

1. **Feature Engineering**
   * Time-based features (year, month, day, day_of_week, quarter)
   * Store and item encoding
   * No lag features to maintain simplicity and reduce data leakage

2. **Model Training**
   * Multiple models implemented:
     * LightGBM Regressor (primary model)
     * Random Forest Regressor
     * Gradient Boosting Regressor
   * Time series cross-validation with 5 folds
   * Hyperparameter optimization using batched grid search:
     * Process parameters in small batches (10 combinations per batch)
     * Save intermediate results after each batch
     * Early stopping for LightGBM (50 rounds)
     * Benefits:
       - Reduces memory usage during optimization
       - Allows for checkpoint recovery if process is interrupted
       - More efficient for large parameter spaces
       - Enables monitoring of optimization progress

3. **Evaluation Metrics**
   * RMSE (Root Mean Square Error)
   * MAE (Mean Absolute Error)
   * R² Score
   * Cross-validation scores with standard deviation

4. **Visualization**
   * Overall sales trends
   * Store-item specific predictions
   * Model performance comparisons

## Usage

1. Open and run `main.ipynb` in Jupyter Notebook/Lab:
```bash
jupyter notebook
```

2. The notebook is structured in sections:
   * Data Loading and Exploration
   * Feature Engineering
   * Model Training and Evaluation
   * Predictions and Visualization

3. Results include:
   * Model performance metrics
   * Visualizations of actual vs predicted sales
   * Future sales predictions for each store-item combination

## Model Performance

The LightGBM model demonstrates strong performance with:
* Time Series Cross-validation to ensure robust predictions
* Separate test set evaluation
* Store-item level accuracy analysis

## Future Improvements

Potential enhancements:
1. Add seasonal features
2. Implement advanced time series techniques
3. Incorporate external factors (holidays, promotions, etc.)
4. Ensemble multiple models
5. Add feature importance analysis

## Requirements

Python version: 3.12

Key dependencies:
* pandas
* numpy
* scikit-learn
* lightgbm
* matplotlib
* seaborn

See `requirements.txt` for complete list. 

__________________________________________________________________________________________________________________

For any questions or issues, please contact [natdanai.sriapai@gmail.com](natdanai.sriapai@gmail.com).