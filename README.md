# ECS171-Group-2-Final---Predicting-Heart-Disease

## Raw Dataset:
Open `heart.csv` file. Also located on the Kaggle website: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

## EDA:
Open `Proj-EDA.ipynb` file.

## Models and Hyperparameter Tuning:
Open `ECS171_Final_LR_SVM_RF_1.ipynb` file.

## Run the website: 
1. Initiate the application by running the following command:
```
python app.py
```
3. After the file successfully runs, a server link will be generated. Then enter the following link into the browser:
```
http://127.0.0.1:8085/
```
## Get prediction from the website

1.Use any date in the dataset `heart.csv` to fill the the attribute values in the website:

- `FastingBS` in datase = `Fasting Blood Suger` in website. If the value of FastingBS in data is `0`, you need to select `> 120 mg/dl` option. If value is `1`, select `<= 120 mg/dl` option.
- `RestingBP` in dataset = `Resting Blood Pressure` in website
- `FastingBS` in datase = `Fasting Blood Suger` in website
- `MaxHr` in dataset = `Maximum Heart Rate` in website
    
2.Click the 'Predict' button. Then the prediction will show on the bottom of website.

## Contributors
- Abhi Gupta
- Brandon Fong
- Xin Tang
- Williyam Yared
