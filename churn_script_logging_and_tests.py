"""
Module contains test of ML churn customer analysis
Author : Max Raphael Sobroza Marques
Date : 21 November 2022
"""
import os
import logging
import pandas as pd
import pytest
from churn_library import (import_data,
                          perform_eda,
                          perform_feature_engineering,
                          encoder_helper,
                          train_models
                          )

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)

def remove_existing_path(file_path: str) -> None:
    """
    Remove a file from the file system if it exists.

    Parameters:
        file_path (str): The path to the file to be removed.

    Returns:
        None
    """
    if os.path.exists(file_path):
        os.remove(file_path)

def test_import():
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        bank_df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert bank_df.shape[0] > 0
        assert bank_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err


# Create a dummy dataframe for testing


@pytest.fixture
def dummy_dataset():
    """dummy"""
    return pd.DataFrame(
        {
            "Attrition_Flag": [
                "Existing Customer",
                "Attrited Customer",
                "Existing Customer",
            ],
            "Customer_Age": [20, 30, 40],
            "Marital_Status": ["Single", "Married", "Single"],
            "Gender": ["M", "M", "F"],
            "Education_Level": ["Master", "Master", "PhD"],
            "Income_Category": [2, 2, 5],
            "Card_Category": [2, 2, 5],
            "Total_Trans_Ct": [2, 3, 4],
        }
    )


@pytest.fixture
def bank_dataset():
    """import bank dataset"""
    return import_data("./data/bank_data.csv")


def test_eda_dummy(dummy_dataset):
    """
    test eda
    """
    remove_existing_path("./images/eda/churn_hist.png")
    remove_existing_path("./images/eda/customer_age_hist.png")
    remove_existing_path("./images/eda/marital_status_bar.png")
    remove_existing_path("./images/eda/total_transaction_dist.png")
    remove_existing_path("./images/eda/correlation_heatmap.png")
    eda_df = perform_eda(dummy_dataset)
    assert isinstance(eda_df, pd.DataFrame)
    assert eda_df.columns.tolist() == [
        "Attrition_Flag",
        "Customer_Age",
        "Marital_Status",
        "Gender",
        "Education_Level",
        "Income_Category",
        "Card_Category",
        "Total_Trans_Ct",
        "Churn",
    ]
    # Check the values of the Churn column
    assert eda_df["Churn"].tolist() == [0, 1, 0]
    # Test that the function creates and saves the expected plots
    assert os.path.exists("./images/eda/churn_hist.png")
    assert os.path.exists("./images/eda/customer_age_hist.png")
    assert os.path.exists("./images/eda/marital_status_bar.png")
    assert os.path.exists("./images/eda/total_transaction_dist.png")
    assert os.path.exists("./images/eda/correlation_heatmap.png")


def test_eda(bank_dataset):
    """
    test eda
    """
    remove_existing_path("./images/eda/churn_hist.png")
    remove_existing_path("./images/eda/customer_age_hist.png")
    remove_existing_path("./images/eda/marital_status_bar.png")
    remove_existing_path("./images/eda/total_transaction_dist.png")
    remove_existing_path("./images/eda/correlation_heatmap.png")
    eda_df = perform_eda(bank_dataset)
    assert isinstance(eda_df, pd.DataFrame)

    # Test that the function creates and saves the expected plots
    assert os.path.exists("./images/eda/churn_hist.png")
    assert os.path.exists("./images/eda/customer_age_hist.png")
    assert os.path.exists("./images/eda/marital_status_bar.png")
    assert os.path.exists("./images/eda/total_transaction_dist.png")
    assert os.path.exists("./images/eda/correlation_heatmap.png")

def test_encoder_helper(bank_dataset):
    """Missing encoder helper"""
    dummy_dataset = perform_eda(bank_dataset)
    category_list = ["Attrition_Flag", "Marital_Status"]
    encoder_helper_df = encoder_helper(dummy_dataset, category_list)

    # Test that the function returns a DataFrame
    assert isinstance(encoder_helper_df, pd.DataFrame)

    # Test that the correct number of new columns is added
    assert len(encoder_helper_df.columns) == len(dummy_dataset.columns) + len(
        category_list
    )

    # Test that the new columns have the correct names
    new_column_names = [f"{cat}_proportion_churn" for cat in category_list]
    assert set(new_column_names).issubset(set(encoder_helper_df.columns))

    # Test that the new columns contain the expected values
    for cat_name in category_list:
        proportion_col_name = f"{cat_name}_proportion_churn"
        cat_group = dummy_dataset.groupby(cat_name).mean()["Churn"]
        for _, row in encoder_helper_df.iterrows():
            assert row[proportion_col_name] == cat_group.loc[row[cat_name]]


def test_perform_feature_engineering(bank_dataset):
    """
    test feature engineering
    """
    x_train, x_test, y_train, y_test = perform_feature_engineering(bank_dataset)

    # Test that the function returns the expected output types
    assert isinstance(x_train, pd.DataFrame)
    assert isinstance(x_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert len(x_train)==len(y_train)
    assert len(x_test)==len(y_test)
    assert len(y_train) > len(y_test)

def test_train_models(bank_dataset):
    """
    test train_models
    """
    remove_existing_path("./models/logistic_model.pkl")
    remove_existing_path("./models/rfc_model.pkl")
    remove_existing_path("./images/results/roc_plot_lr.png")
    remove_existing_path("./images/results/roc_plot_rfc.png")
    remove_existing_path("./images/results/lr_train_report.txt")
    remove_existing_path("./images/results/lr_test_report.txt")
    remove_existing_path("./images/results/rf_train_report.txt")
    remove_existing_path("./images/results/rf_test_report.txt")
    remove_existing_path("./images/results/lr_feature_importance.png")
    remove_existing_path("./images/results/rfc_feature_importance.png")
    remove_existing_path("./images/results/lr_feature_importance_shap.png")
    remove_existing_path("./images/results/rfc_feature_importance_shap.png")
    train_models(*perform_feature_engineering(bank_dataset.sample(100)))
    assert os.path.exists("./models/logistic_model.pkl")
    assert os.path.exists("./models/rfc_model.pkl")
    assert os.path.exists("./images/results/roc_plot_lr.png")
    assert os.path.exists("./images/results/roc_plot_rfc.png")
    assert os.path.exists("./images/results/lr_train_report.txt")
    assert os.path.exists("./images/results/lr_test_report.txt")
    assert os.path.exists("./images/results/rf_train_report.txt")
    assert os.path.exists("./images/results/rf_test_report.txt")
    assert os.path.exists("./images/results/lr_feature_importance.png")
    assert os.path.exists("./images/results/rfc_feature_importance.png")
    assert os.path.exists("./images/results/rfc_feature_importance_shap.png")


if __name__ == "__main__":
    pass
