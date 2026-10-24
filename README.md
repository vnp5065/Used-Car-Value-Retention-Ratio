# Used-Car-Value-Retention-Ratio
# Predicting Used Car Value Retention Ratio Using Machine Learning

## Project Overview

This project uses machine learning to predict the value retention ratio of used vehicles based on their characteristics. Vehicle listing data was collected from two websites, Everylot and SeeAllCars, using Python web scraping.

The value retention ratio is calculated as:

**Value Retention Ratio = Current Listing Price / Original MSRP**

## Dataset

The project includes 1,010 cleaned vehicle listings from three manufacturers: Toyota, Honda, and Ford. Historical MSRP information was matched with 117 vehicles for machine learning.

## Machine Learning Models

Three regression models were implemented:

- Linear Regression
- Random Forest Regression
- Gradient Boosting Regression

Random Forest achieved the lowest RMSE of 0.2747, while Linear Regression achieved the lowest MAE of 0.1573. All three models had negative R² scores, indicating limitations in prediction accuracy.

## Tools and Libraries

Python, Pandas, NumPy, BeautifulSoup, Requests, Scikit-learn, Matplotlib, and Seaborn.

## Project Files

The repository contains Python scripts for web scraping, data preprocessing, MSRP matching, feature engineering, and machine learning. It also includes CSV datasets, model evaluation results, graphs, and the final project report.

## Conclusion

This project demonstrates how web scraping, data preprocessing, feature engineering, and regression models can be combined to analyze used vehicle depreciation. The main challenge was matching historical MSRP information with vehicle listings, which limited the number of samples available for model training.
