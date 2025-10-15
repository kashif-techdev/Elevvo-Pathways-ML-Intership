# Sales Forecasting Project

## What This Project Does

This project predicts future sales using machine learning. It includes all the bonus features required for the assignment.

## What You Get

- **Basic Models**: Linear Regression, Random Forest
- **Advanced Models**: XGBoost, LightGBM (bonus features)
- **Smart Features**: Rolling averages, seasonal patterns (bonus features)
- **Proper Testing**: Time-aware validation (bonus feature)
- **Visual Results**: Charts and graphs showing how well the models work

## Files in This Project

- `sales_forecasting.py` - Main program
- `sales_forecasting_analysis.ipynb` - Interactive notebook
- `requirements.txt` - What to install
- `sales_forecasting_results.png` - Results chart

## How to Run

### Step 1: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 2: Run the Program
```bash
python sales_forecasting.py
```

## What the Program Does

1. **Creates Sample Data**: Makes fake Walmart-like sales data
2. **Adds Smart Features**: 
   - Time features (month, week, etc.)
   - Previous sales (lag features)
   - Rolling averages (bonus)
   - Seasonal patterns (bonus)
3. **Trains Models**: 4 different machine learning models
4. **Tests Models**: Uses proper time-based testing
5. **Shows Results**: Creates charts and performance numbers

## Sample Data Used

- 5 stores
- 3 departments per store  
- 150 weeks of data
- Includes holidays, weather, and economic data

## Model Results

The program shows how well each model predicts sales:

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | 0.00 | 0.00 | 1.0000 |
| Random Forest | 1353.33 | 1605.17 | 0.7408 |
| XGBoost | 1256.09 | 1516.43 | 0.7687 |
| LightGBM | 1148.99 | 1402.84 | 0.8020 |

## Bonus Features Included

✅ **Rolling Averages**: Uses past 3 and 7 weeks to predict future sales
✅ **Seasonal Patterns**: Finds yearly patterns in sales data  
✅ **Advanced Models**: XGBoost and LightGBM for better predictions
✅ **Smart Testing**: Uses time-based validation to avoid cheating

## What You Need

- Python 3.8 or newer
- All packages listed in requirements.txt

## How to Use Real Data

To use actual Walmart data instead of fake data:

1. Download Walmart sales data from Kaggle
2. Put the CSV files in this folder
3. Change the data loading part in the code

## Results

The program creates a picture file called `sales_forecasting_results.png` that shows:
- How well each model predicts sales
- Which model works best
- What features are most important

## Screen shot
<img width="1284" height="935" alt="{4FF94DDF-8FF7-433F-AAC5-48E8AD3913E3}" src="https://github.com/user-attachments/assets/3454c372-c4b8-43e9-b5fc-d287bb100f69" />


## Summary

This project successfully creates a sales forecasting system with all required bonus features. It's ready to use and can be easily modified for different data or requirements.
