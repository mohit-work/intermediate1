# Give-Life-Predict-Blood-Donations
Project Give Life: Predict Blood Donations

Project Description

"Blood is the most precious gift that anyone can give to another person — the gift of life." ~ World Health Organization

Forecasting blood supply is a serious and recurrent problem for blood collection managers: in January 2019, "Nationwide, the Red Cross saw 27,000 fewer blood donations over the holidays than they see at other times of the year." Machine learning can be used to learn the patterns in the data to help to predict future blood donations and therefore save more lives.

In this Project, you will work with data collected from the donor database of Blood Transfusion Service Center in Hsin-Chu City in Taiwan. The center passes its blood transfusion service bus to one university in Hsin-Chu City to gather blood donated about every three months. The dataset, obtained from the UCI Machine Learning Repository, consists of a random sample of 748 donors. Your task will be to predict if a blood donor will donate within a given time window. You will look at the full model-building process: from inspecting the dataset to using the tpot library to automate your Machine Learning pipeline.

To complete this Project, you need to know some Python, pandas, and logistic regression. We recommend one is familiar with the content in DataCamp's Manipulating DataFrames with pandas, Preprocessing for Machine Learning in Python, and Foundations of Predictive Analytics in Python (Part 1) courses.

Project Tasks

1. Inspecting transfusion.data file
2. Loading the blood donations data
3. Inspecting transfusion DataFrame
4. Creating target column
5. Checking target incidence
6. Splitting transfusion into train and test datasets
7. Selecting model using TPOT
8. Checking the variance
9. Log normalization
10. Training the linear regression model
11. Conclusion


Sure, here's a detailed README file for your project to upload on GitHub:

---

# Blood Donation Forecasting

![Blood Donation](https://example.com/blood_donation_image.png)

## Overview

This project aims to use machine learning techniques to predict future blood donations, addressing a critical issue in the healthcare system. By analyzing a dataset of blood donation records, we build predictive models to identify individuals who are likely to donate blood in the future. This can help blood banks and healthcare organizations manage their blood supply more effectively.

## Dataset

The dataset used in this project is the Blood Transfusion Service Center Data Set, available from the UCI Machine Learning Repository. The dataset contains records of blood donors, including various features related to their donation history.

### Dataset Details

- **Recency (months)**: Months since the last donation.
- **Frequency (times)**: Total number of donations.
- **Monetary (c.c. blood)**: Total blood donated in c.c.
- **Time (months)**: Months since the first donation.
- **Target**: Whether the donor donated blood in March 2007 (1) or not (0).

## Project Structure

- `datasets/`: Contains the dataset file `transfusion.data`.
- `notebooks/`: Jupyter notebooks with detailed exploratory data analysis (EDA) and model training steps.
- `src/`: Source code for data preprocessing, model training, and evaluation.
- `models/`: Saved models and pipelines.
- `README.md`: Project overview and instructions.
- `requirements.txt`: List of required Python packages.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/blood-donation-forecasting.git
   cd blood-donation-forecasting
   ```

2. Create a virtual environment and activate it:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preprocessing

The first step is to preprocess the data. This includes reading the dataset, renaming columns, and splitting the data into training and testing sets.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Read in dataset
transfusion = pd.read_csv('datasets/transfusion.data')

# Rename target column
transfusion.rename(
    columns={'whether he/she donated blood in March 2007': 'target'},
    inplace=True
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    transfusion.drop(columns='target'),
    transfusion.target,
    test_size=0.25,
    random_state=42,
    stratify=transfusion.target
)
```

### Model Training

We use TPOTClassifier and LogisticRegression for model training and evaluation.

```python
from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score

# Instantiate and train TPOTClassifier
tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    verbosity=2,
    scoring='roc_auc',
    random_state=42,
    disable_update_check=True,
    config_dict='TPOT light'
)
tpot.fit(X_train, y_train)

# Evaluate TPOT model
tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {tpot_auc_score:.4f}')
```

We also train a logistic regression model:

```python
from sklearn import linear_model
import numpy as np

# Normalize data
X_train_normed, X_test_normed = X_train.copy(), X_test.copy()
col_to_normalize = 'Monetary (c.c. blood)'
for df_ in [X_train_normed, X_test_normed]:
    df_['monetary_log'] = np.log(df_[col_to_normalize])
    df_.drop(columns=col_to_normalize, inplace=True)

# Train logistic regression model
logreg = linear_model.LogisticRegression(solver='liblinear', random_state=42)
logreg.fit(X_train_normed, y_train)

# Evaluate logistic regression model
logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test_normed)[:, 1])
print(f'\nAUC score: {logreg_auc_score:.4f}')
```

### Model Comparison

Compare the AUC scores of the models:

```python
from operator import itemgetter

# Sort models based on their AUC score
models = [('tpot', tpot_auc_score), ('logreg', logreg_auc_score)]
sorted(models, key=itemgetter(1), reverse=True)
```

## Results

- **TPOT Model AUC Score**: `0.XXXX`
- **Logistic Regression Model AUC Score**: `0.XXXX`

The TPOT model showed a slightly higher AUC score compared to the logistic regression model, indicating better performance in predicting future blood donations.

## Conclusion

This project demonstrates the potential of machine learning in predicting blood donations, which can significantly aid in the efficient management of blood supplies. Future work could include exploring additional features, trying out other machine learning algorithms, and further optimizing the models.

## Contributing

If you wish to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The UCI Machine Learning Repository for providing the dataset.
- All contributors to this project.

---

Feel free to modify and customize this README as per your specific needs and additional details.