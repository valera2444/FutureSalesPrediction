# Fututre sales prediction

## Introduction

Future sales prediction for the cartesian product of known shops and items for the next month. Project provides opportunity to run full pipeline from raw data to the final predictions.

## Dependencies

`requirements.txt` gives the default packages required.

Main dependencies are:

- Python3.12
- numpy, `pip install numpy`
- pandas, `pip install pandas`
- scikit-learn, `pip install scikit-learn`
- lightgbm, `pip install lightgbm`
- optuna `pip install optuna`

> **Note**: Dependencies must be installed manually or via requirements.txt. 

## Install

- from [Test PyPI](https://test.pypi.org/project/predict-future-sales)


## Run

Example usage:

```python
from future_sales_prediction import etl, prepare_data, create_submission, hyperparameters_tuning

data_source = 'data'
data_etl_writes_to = 'data/cleaned'
data_prepare_data_writes_to = 'data'

etl.run_etl(data_source,data_etl_writes_to)

prepare_data.run_prepare_data(data_etl_writes_to,data_prepare_data_writes_to)

params = hyperparameters_tuning.run_optimizing(data_prepare_data_writes_to, data_etl_writes_to, n_trials=2)

create_submission.run_create_submission(data_prepare_data_writes_to, data_etl_writes_to, params)
```
Here user-defined parameters are:
- Directory with source data and directories for storing temporary data.
- Number of trials for hyperparameter optimization. Note, that one trial takes around 20 minutes.

## Extract-transform-load `(etl.py)`
This part handles inconsistent data

## Feature creation `(prepare_data.py)`
This stage creates data for training and inference. After running this stage we will have large .csv file, with rows representing (shop, item) pairs and columns representing:
- **Monthly Features**: Columns related to the month-specific characteristics of each (shop, item) pair. These features are tagged with a date suffix in the format `{feature_name}$dbn`, where `dbn` is the `date_block_num` (e.g., `ema_item_category_id$10`).
- **Static Features**: Columns that are constant over time, such as static attributes of shops or items.
> **Note**: Auxillary csv files created during running this stage in a folder defined by user. 


## Data final preparations, model training and submission creation `(create_submission.py)`:
This stage:
1) Runs final data preparation to make data suitable for model fitting and inferece. Feature selection step takes place on this stage.
2) Trains model and creates submission. Submission is written into submission.csv in a root directory. 

## Hyperparameter tuning `(hyperparameter_tuning.py)`:
This stage:
1) Is very similar to the previous one, but additionally iterates over hyperparameter space searching for the best hyperparameters.