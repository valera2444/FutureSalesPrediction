# FutureSalesPrediction
Short explanation of files and dependecies between them

./EDA/EDA_transorm/ipynb reads from ./data_cleaned/* for creating csv file which is saved as ./data.csv

./validation/prepare_data_*  Reads from ./data_cleaned/* and ./data.csv for creating ./validaion/data/merged.csv file

./validation/validation_as_mix_* reads ./validation/data/merged.csv and ./data_cleaned/* for performing validation. Also submission created here and saved as ./validation/submission.csv

Before running validation_as_mix_boosting and validation_as_mix_LSTM  it's recommended to run prepare_data_as_MIX_boosting and prepare_data_as_MIX_LSTM respectively. You can choose columns that you will use during validation by changing "names" array. Also consider, that using too large number of columns may not fit into memory(both RAM and GPU).
 
 
