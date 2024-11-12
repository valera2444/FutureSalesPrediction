from future_sales_prediction import prepare_data, etl, create_submission, hyperparameters_tuning

data_source = 'data'
data_etl_writes_to = 'data/cleaned'
data_prepare_data_writes_to = 'data'

etl.run_etl(data_source,data_etl_writes_to)

prepare_data.run_prepare_data(data_etl_writes_to,data_prepare_data_writes_to)

params = hyperparameters_tuning.run_optimizing(data_prepare_data_writes_to, data_etl_writes_to, n_trials=1)

create_submission.run_create_submission(data_prepare_data_writes_to, data_etl_writes_to, params)