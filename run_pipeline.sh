#!/usr/bin/env bash

run_name="run_1"

/home/valeriy/python_envs/predict_future_sales/bin/python3.12 /home/valeriy/python_projects/predict_future_sales/future_sales_prediction/etl.py --run_name ${run_name} --source_path ${run_name}/data --destination_path ${run_name}/data/cleaned

/home/valeriy/python_envs/predict_future_sales/bin/python3.12 /home/valeriy/python_projects/predict_future_sales/future_sales_prediction/prepare_data.py --run_name ${run_name} --source_path  ${run_name}/data/cleaned --destination_path  ${run_name}/data

/home/valeriy/python_envs/predict_future_sales/bin/python3.12 /home/valeriy/python_projects/predict_future_sales/future_sales_prediction/hyperparameters_tuning.py --run_name ${run_name} --path_for_merged  ${run_name}/data --path_data_cleaned   ${run_name}/data/cleaned --n_trials  1

/home/valeriy/python_envs/predict_future_sales/bin/python3.12 /home/valeriy/python_projects/predict_future_sales/future_sales_prediction/create_submission.py --run_name ${run_name} --path_for_merged ${run_name}/data --path_data_cleaned  ${run_name}/data/cleaned --is_create_submission 0

/home/valeriy/python_envs/predict_future_sales/bin/python3.12 /home/valeriy/python_projects/predict_future_sales/future_sales_prediction/create_submission.py --run_name ${run_name} --path_for_merged ${run_name}/data --path_data_cleaned  ${run_name}/data/cleaned --is_create_submission 1

/home/valeriy/python_envs/predict_future_sales/bin/python3.12 /home/valeriy/python_projects/predict_future_sales/future_sales_prediction/error_analysis.py --run_name ${run_name} --path_for_merged ${run_name}/data --path_data_cleaned ${run_name}/data/cleaned --path_artifact_storage ${run_name}/images

/home/valeriy/python_envs/predict_future_sales/bin/python3.12 /home/valeriy/python_projects/predict_future_sales/future_sales_prediction/model_interpret.py --run_name ${run_name} --path_for_merged ${run_name}/data --path_data_cleaned ${run_name}/data/cleaned --path_artifact_storage ${run_name}/images
