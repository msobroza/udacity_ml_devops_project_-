# Author

This project was done by Max Raphael Sobroza Marques in 8th January 2023

# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This is my first project of course ML DevOps Engineer Nanodegree Udacity
The goal of this project is to refactor a Notebook python to script a python code

## Files and data description
Overview of the files and data present in the root directory.

* Root directory
  * data
    * `bank_data.csv` # data
  * images # It contains all images from all preprocessing steps
    * eda
    * results
  * models # Stored models
  * `churn_library.py` # File that contains all functions of EDA, preprocessing and to train a ML model
  * `churn_script_logging_and_tests.py` # Test file
## Running Files
1. Install libraries

```
# Workspace run
python -m pip install -r requirements_py3.6.txt
```

2. Test the code

`pylint churn_script_logging_and_tests.py`

3. Execute the entire pipeline

`python churn_script.py`
