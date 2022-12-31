"""
Module contain functions of ML churn customer analysis
Author : Max Raphael Sobroza Marques
Date : 21 November 2022
"""

# library doc string
# import libraries
# Standard library imports
import os
from typing import Any, Dict, List, Optional, Type, Union
from pathlib import Path
import matplotlib.pyplot as plt

# Third-party library imports
import numpy as np
import pandas as pd
import seaborn as sns
import shap

# Sklearn imports
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

# Custom imports
import joblib

sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Categorical features
CAT_COLUMNS = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]
# Numerical features
QUANT_COLUMNS = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]
# Features to use in classifier
KEEP_COLS = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
    "Gender_Churn",
    "Education_Level_Churn",
    "Marital_Status_Churn",
    "Income_Category_Churn",
    "Card_Category_Churn",
]
CHURN_COLUMN = "Churn"
ATTRITION_FLAG_MAP = {"Existing Customer": 0, "Attrited Customer": 1}

# Plot sizes definition
EDA_FIGURE_SIZE = (20, 10)
ROC_PLOT_FIGURE_SIZE = (15, 8)

# Paths definition
EDA_PLOT_RELATIVE_PATH = "./images/eda/"
RESULTS_PLOT_RELATIVE_PATH = "./images/results/"
MODEL_RELATIVE_PATH = "./models"

# Create directories if does not exists
for dir_name in [EDA_PLOT_RELATIVE_PATH, RESULTS_PLOT_RELATIVE_PATH, MODEL_RELATIVE_PATH]:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def encode_churn(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'Churn' column to a dataframe with a numerical value representing
    the 'Attrition_Flag' column.

    Parameters:
        input_df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: A modified version of the input dataframe with a new
            'Churn' column added.
    """
    output_df = input_df.copy(deep=True)
    output_df[CHURN_COLUMN] = input_df["Attrition_Flag"].apply(
            lambda val: ATTRITION_FLAG_MAP[val]
        )
    return output_df

def import_data(path: Path) -> pd.DataFrame:
    """
    Import a CSV file and return the resulting Pandas dataframe.

    Parameters:
        path (Path): The path to the CSV file.

    Returns:
        output_input_df (pd.DataFrame): The Pandas dataframe created from the CSV file.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {str(path)} does not exists")
    return pd.read_csv(path)

def perform_eda(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform eda on input_df and save figures to images folder

    Parameters:
            input_df (pd.DataFrame): DataFrame with samples

    Returns:
            eda_df (pd.DataFrame): result of eda
    """
    # Copy dataframe
    eda_df = input_df.copy(deep=True)

    eda_df = encode_churn(eda_df)

    # Plot Churn histogram
    _, ax_fig = plt.subplots(figsize=EDA_FIGURE_SIZE)
    eda_df.hist("Churn", ax=ax_fig)
    plt.savefig(os.path.join(EDA_PLOT_RELATIVE_PATH, "churn_hist.png"))

    # Plot Customer Age histogram
    plt.plot(figsize=EDA_FIGURE_SIZE)
    eda_df.hist("Customer_Age", ax=ax_fig)
    plt.savefig(os.path.join(EDA_PLOT_RELATIVE_PATH, "customer_age_hist.png"))

    # Plot Marital Status bar plot
    plt.plot(figsize=EDA_FIGURE_SIZE)
    eda_df.Marital_Status.value_counts("normalize").plot(kind="bar")
    plt.savefig(os.path.join(EDA_PLOT_RELATIVE_PATH, "marital_status_bar.png"))

    # Plot Total Transaction distribution
    plt.plot(figsize=EDA_FIGURE_SIZE)
    sns.distplot(eda_df["Total_Trans_Ct"])
    plt.savefig(
        os.path.join(
            EDA_PLOT_RELATIVE_PATH,
            "total_transaction_dist.png"))

    # Plot Correlation heatmap
    plt.plot(figsize=EDA_FIGURE_SIZE)
    sns.heatmap(eda_df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig(
        os.path.join(
            EDA_PLOT_RELATIVE_PATH,
            "correlation_heatmap.png"))

    # Returns the EDA DataFrame
    return eda_df


def encoder_helper(
    input_df: pd.DataFrame,
    category_list: List[str],
    response: Optional[str] = "proportion_churn",
) -> pd.DataFrame:
    """
    Add new columns to a dataframe with the mean value of
        the 'Churn' column for each categorical value.

    Parameters:
        input_df (pd.DataFrame): The input dataframe.
        category_list (List[str]): A list of columns in 'input_df'
            that contain categorical values.
        response (Optional[str]): The name of the response column.
        If not provided, defaults to 'proportion_churn'.

    Returns:
        pd.DataFrame: A modified version of the input dataframe with new
            columns added for each categorical column in 'category_list'.
    """
    # Copy the dataframe
    enc_helper_df = input_df.copy(deep=True)
    enc_helper_df = encode_churn(enc_helper_df)
    for cat_name in category_list:
        enc_cat_group = enc_helper_df.groupby(cat_name).mean()[CHURN_COLUMN]
        enc_cat_list = [
            enc_cat_group.loc[enc_cat_value]
            for enc_cat_value in enc_helper_df[cat_name]
        ]
        proportion_col_name = f"{cat_name}_{response}"
        enc_helper_df[proportion_col_name] = enc_cat_list
    return enc_helper_df

def perform_feature_engineering(
    input_df: pd.DataFrame, response: Optional[str] = CHURN_COLUMN
):
    """Perform feature engineering on a given dataframe and split
    it into training and testing sets.

    Parameters:
        input_df (pd.DataFrame): The input dataframe.
        response (Optional[str]): The name of the response column.
        If not provided, defaults to 'proportion_churn'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing
        the training and testing data for the input
        features ('x_train', 'x_test') and the response ('y_train', 'y_test').
    """
    df_filtered = input_df.copy(deep=True)
    df_filtered = encode_churn(df_filtered)
    encoded_df = encoder_helper(df_filtered, CAT_COLUMNS, response)
    encoded_x = encoded_df[KEEP_COLS]
    encoded_y = encoded_df[CHURN_COLUMN]
    return train_test_split(
        encoded_x, encoded_y, test_size=0.3, random_state=42
    )


def classification_report_image(
    y_train: pd.Series,
    y_test: pd.Series,
    y_train_preds_lr: pd.Series,
    y_train_preds_rf: pd.Series,
    y_test_preds_lr: pd.Series,
    y_test_preds_rf: pd.Series,
) -> None:
    """
    Generate and save classification reports for training and testing results.

    Parameters:
        y_train (pd.Series): Training response values.
        y_test (pd.Series): Test response values.
        y_train_preds_lr (pd.Series): Training predictions from logistic regression.
        y_train_preds_rf (pd.Series): Training predictions from random forest.
        y_test_preds_lr (pd.Series): Test predictions from logistic regression.
        y_test_preds_rf (pd.Series): Test predictions from random forest.

    Returns:
        None
    """
    # Generate and save classification reports
    lr_train_report = classification_report(y_train, y_train_preds_lr)
    lr_test_report = classification_report(y_test, y_test_preds_lr)
    rf_train_report = classification_report(y_train, y_train_preds_rf)
    rf_test_report = classification_report(y_test, y_test_preds_rf)
    # Save reports in a test file
    with open(os.path.join(RESULTS_PLOT_RELATIVE_PATH,
                           "lr_train_report.txt"), "w") as file:
        file.write(lr_train_report)
    with open(os.path.join(RESULTS_PLOT_RELATIVE_PATH,
                           "lr_test_report.txt"), "w") as file:
        file.write(lr_test_report)
    with open(os.path.join(RESULTS_PLOT_RELATIVE_PATH,
                           "rf_train_report.txt"), "w") as file:
        file.write(rf_train_report)
    with open(os.path.join(RESULTS_PLOT_RELATIVE_PATH,
                           "rf_test_report.txt"), "w") as file:
        file.write(rf_test_report)


def feature_importance_plot(
    model: Union[LogisticRegression, RandomForestClassifier],
    x_data: pd.DataFrame,
    output_pth: str,
) -> None:
    """
    Plot and save the feature importances of a given model.

    Parameters:
        model (Union[LogisticRegression, RandomForestClassifier]):
            The model to generate feature importances for.
        x_data (pd.DataFrame): The input data for the model.
        output_pth (str): The path to save the plot image.

    Returns:
        None
    """
    output_pth = os.path.join(RESULTS_PLOT_RELATIVE_PATH, output_pth)
    # Extract and plot feature importances
    if isinstance(model, LogisticRegression):
        coef = model.coef_
        feature_importances = abs(coef[0])
    else:
        feature_importances = model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    feature_names = x_data.columns[sorted_idx]
    plt.figure(figsize=(10, 5))
    plt.bar(range(x_data.shape[1]), feature_importances[sorted_idx])
    plt.xticks(range(x_data.shape[1]), feature_names, rotation=90)
    plt.title("Feature Importances")
    plt.savefig(output_pth)


def feature_importance_shap_plot(
    model: Union[LogisticRegression, RandomForestClassifier],
    x_data: pd.DataFrame,
    output_pth: str,
) -> None:
    """
    Plot and save the feature importances of a given model.

    Parameters:
        model (Union[LogisticRegression, RandomForestClassifier]): The model to
            generate feature importances for.
        x_data (pd.DataFrame): The input data for the model.
        output_pth (str): The path to save the plot image.

    Returns:
        None
    """
    output_pth = os.path.join(RESULTS_PLOT_RELATIVE_PATH, output_pth)
    # Calculate feature importances using shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_data)
    plt.figure(figsize=EDA_FIGURE_SIZE)
    shap.summary_plot(shap_values, x_data)
    plt.savefig(output_pth)


def plot_save_roc_curve(
        model: BaseEstimator,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        plot_file_path: str) -> None:
    """
    Generate a ROC curve plot for a given model and test data, and store it as an image.

    Parameters:
        model (BaseEstimator): The model to generate the ROC curve for.
        x_test (pd.DataFrame): The input data for the model.
        y_test (pd.Series): The true labels for the input data.
        plot_file_path (str): The file path to save the plot image.

    Returns:
        None
    """
    path = os.path.join(RESULTS_PLOT_RELATIVE_PATH, plot_file_path)
    plt.figure(figsize=ROC_PLOT_FIGURE_SIZE)
    plot_roc_curve(model, x_test, y_test)
    plt.savefig(path)


def save_model(model: BaseEstimator, model_file_path: str) -> None:
    """
    Save a trained model to a file.

    Parameters:
        model (BaseEstimator): The model to save.
        model_file_path (str): The file path to save the model to.

    Returns:
        None
    """
    path = os.path.join(MODEL_RELATIVE_PATH, model_file_path)
    joblib.dump(model, path)


def load_model(model_file_path: str) -> BaseEstimator:
    """
    Load a trained model from a file.

    Parameters:
        model_file_path (str): The file path of the model to load.

    Returns:
        BaseEstimator: The loaded model.
    """
    path = os.path.join(MODEL_RELATIVE_PATH, model_file_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model filepath: {path} does not exists")
    return joblib.load(path)


def store_classification_report_plot(
    y_true: pd.Series, y_pred: pd.Series, title: str, output_path: str
) -> None:
    """
    Generate a classification report plot and store it as an image.

    Parameters:
        y_true (pd.Series): The true labels.
        y_pred (pd.Series): The predicted labels.
        title (str): The title of the plot.
        output_path (str): The path to save the plot image.

    Returns:
        None
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.plot.bar()
    plt.title(title)
    plt.savefig(output_path)


def train_model_grid_search(
    model: Type[BaseEstimator],
    param_grid: Dict[str, List[Any]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> BaseEstimator:
    """
    Train a model using a grid search to find the best hyperparameters.

    Parameters:
        model (Type[BaseEstimator]): The model class to use.
        param_grid (Dict[str, List[Any]]): The hyperparameter grid to search over.
        x_train (pd.DataFrame): Training data for the input features.
        y_train (pd.Series): Training data for the response.

    Returns:
        BaseEstimator: The best model found by the grid search.
    """
    cv_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    cv_model.fit(x_train, y_train)
    return cv_model.best_estimator_


def train_random_forest_grid_search(
    x_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestClassifier:
    """
    Train a random forest model using a grid search to find the best hyperparameters.

    Parameters:
        x_train (pd.DataFrame): Training data for the input features.
        y_train (pd.Series): Training data for the response.

    Returns:
        RandomForestClassifier: The best random forest model found by the grid search.
    """
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }

    return train_model_grid_search(
        model=rfc, param_grid=param_grid, x_train=x_train, y_train=y_train
    )


def train_logistic_regression_grid_search(
    x_train: pd.DataFrame,
    y_train: pd.Series
) -> LogisticRegression:
    """
    Train a logistic regression model using a
    grid search to find the best hyperparameters.

    Parameters:
        x_train (pd.DataFrame): Training data for the input features.
        y_train (pd.Series): Training data for the response.

    Returns:
        LogisticRegression: The best logistic regression model found by the grid search.
    """
    lr_model = LogisticRegression(random_state=42)
    param_grid = {
        "penalty": ["l1", "l2"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "solver": ["liblinear", "lbfgs", "saga"],
    }

    return train_model_grid_search(
        model=lr_model, param_grid=param_grid, x_train=x_train, y_train=y_train
    )


def train_models(
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series) -> None:
    """
    Train and save logistic regression and random forest models,
    and generate ROC curve plots and a classification report plot.

    Parameters:
        x_train (pd.DataFrame): The training input data.
        x_test (pd.DataFrame): The testing input data.
        y_train (pd.Series): The training true labels.
        y_test (pd.Series): The testing true labels.

    Returns:
        None
    """
    # Train logistic regression and random forest models using grid search
    best_lr = train_logistic_regression_grid_search(
        x_train=x_train, y_train=y_train)
    best_rfc = train_random_forest_grid_search(
        x_train=x_train, y_train=y_train)

    # Save the trained models
    save_model(best_lr, "logistic_model.pkl")
    save_model(best_rfc, "rfc_model.pkl")

    # Generate ROC curve plots for the trained models
    plot_save_roc_curve(
        model=best_lr,
        x_test=x_test,
        y_test=y_test,
        plot_file_path="roc_plot_lr.png")

    plot_save_roc_curve(
        model=best_rfc,
        x_test=x_test,
        y_test=y_test,
        plot_file_path="roc_plot_rfc.png")
    # Generate a classification report plot for the trained models
    classification_report_image(
        y_train=y_train,
        y_test=y_test,
        y_train_preds_lr=best_lr.predict(x_train),
        y_test_preds_lr=best_lr.predict(x_test),
        y_train_preds_rf=best_rfc.predict(x_train),
        y_test_preds_rf=best_rfc.predict(x_test),
    )
    # Save plots of feature importance
    feature_importance_plot(
        best_lr, x_data=x_test, output_pth="lr_feature_importance.png"
    )
    feature_importance_plot(
        best_rfc, x_data=x_test, output_pth="rfc_feature_importance.png"
    )
    feature_importance_shap_plot(
        best_rfc, x_data=x_test, output_pth="rfc_feature_importance_shap.png"
    )


if __name__ == "__main__":
    data_df = import_data("./data/bank_data.csv")
    _ = perform_eda(data_df)
    train_models(*perform_feature_engineering(data_df))
