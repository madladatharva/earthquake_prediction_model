# Earthquake Prediction System

## Installation Instructions

### Prerequisites
- Python 3.7 or higher
- Required libraries: `numpy`, `pandas`, `scikit-learn`, etc.

### Steps to Install Dependencies
1. Clone the repository:
   ```bash
   git clone https://github.com/madladatharva/earthquake_prediction_model.git
   ```
2. Navigate to the project directory:
   ```bash
   cd earthquake_prediction_model
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Examples

To run the earthquake prediction model, execute the following command:
```bash
python predict.py --input data/input_data.csv
```
The output will be generated in the `results/` directory.

## Features
- Predicts potential earthquake occurrences based on historical data.
- Provides visualization of prediction results.

## Model Performance
- Accuracy: 90%
- Precision: 85%
- Recall: 80%

## Troubleshooting
- If you encounter a `ModuleNotFoundError`, ensure all dependencies are installed correctly.
- For issues related to data formats, double-check the input data structure.
