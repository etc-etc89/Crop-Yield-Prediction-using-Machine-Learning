Crop Yield Prediction using Machine Learning
A data-driven project to forecast agricultural crop yields in India using a Multiple Linear Regression model. This repository contains the Python script, datasets, and documentation for a model that leverages historical data on weather, soil quality, and farming practices to provide reliable yield predictions.

📋 Table of Contents
Project Overview

Model Performance

Datasets Used

Methodology

How to Run the Project

Results and Visualization

Limitations and Future Work

🎯 Project Overview
The agricultural sector faces significant uncertainty in predicting crop yields, which leads to inefficiencies in storage, pricing, and distribution. This project addresses this challenge by building a machine learning model to provide accurate, data-driven forecasts. By analyzing the complex relationships between various environmental and agricultural factors, the model empowers farmers and stakeholders to make informed decisions, reduce post-harvest losses, and enhance financial stability.

📈 Model Performance
The Multiple Linear Regression model was trained and evaluated on a comprehensive dataset, demonstrating strong predictive capabilities.

R-squared (R 
2
 ) Score: 0.78

An R-squared score of 0.78 indicates that 78% of the variance in crop yield can be explained by the features in our model (weather, soil nutrients, fertilizer, etc.). This is a robust result for a complex, real-world problem and confirms the model's effectiveness.

🗂️ Datasets Used
The model was trained on a master dataset created by merging three distinct, real-world data sources:

crop_yield.csv: The core dataset containing historical records of crop production, area, fertilizer/pesticide usage, and yield from 1997 onwards for various states in India.

state_soil_data.csv: Provides state-specific soil profiles, including levels of Nitrogen (N), Phosphorus (P), Potassium (K), and pH.

state_weather_data_1997_2020.csv: Contains historical annual weather data for each state, including average temperature, total rainfall, and average humidity.

⚙️ Methodology
The project follows a standard data science pipeline:

Data Loading and Merging: The three source CSV files are loaded into pandas DataFrames and merged into a single master dataset based on common state and year columns.

Data Cleaning: Rows with any missing values resulting from the merge are dropped to ensure data integrity for model training.

Preprocessing: Categorical features like crop, season, and state are converted into a numerical format using one-hot encoding, making them suitable for the regression model.

Model Training: The dataset is split into an 80% training set and a 20% testing set. A Multiple Linear Regression model is trained on the training data.

Evaluation: The trained model's performance is evaluated on the unseen test set using metrics like R-squared, MSE, and RMSE.

▶️ How to Run the Project
To replicate the project and run the model on your local machine, follow these steps:

1. Prerequisites:

Python 3.7+

The required Python libraries are listed in requirements.txt.

2. Clone the Repository:

git clone [https://github.com/YOUR_USERNAME/your-repo-name.git](https://github.com/YOUR_USERNAME/your-repo-name.git)
cd your-repo-name

3. Install Dependencies:
Create a file named requirements.txt with the following content:

pandas
numpy
scikit-learn
matplotlib

Then, install the libraries using pip:

pip install -r requirements.txt

4. Place Datasets:
Ensure the three data files (crop_yield.csv, state_soil_data.csv, and state_weather_data_1997_2020.csv) are in the same directory as the Python script.

5. Run the Script:
Execute the main script from your terminal:

python crop_yield_prediction.py

The script will process the data, train the model, print the performance metrics to the console, and display the results visualization plot.

📊 Results and Visualization
Upon execution, the script will output the model's performance metrics. The primary visual output is a scatter plot comparing the model's predictions against the actual values from the test set. The tight clustering of points around the red diagonal line is a clear indicator of the model's accuracy.

🔮 Limitations and Future Work
While the model performs well, there are several avenues for improvement:

Data Granularity: The current model uses state-level averages. Using district-level or farm-level data could capture local variations and significantly improve accuracy.

Advanced Models: Experimenting with more complex, non-linear models (e.g., Gradient Boosting, Random Forests) could potentially capture more intricate patterns in the data.

Additional Features: Incorporating more features like satellite imagery, extreme weather event flags, irrigation data, and seed quality could further enhance the model's predictive power.
