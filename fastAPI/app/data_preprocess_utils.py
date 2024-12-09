
import numpy as np

from collections import defaultdict
def select_columns_for_reading(row, dbn):#THIS FUNCTION MUUUUUST MATCH THE ONE IN utils.py
    """
    Selects relevant columns to read from a CSV file based on the date block number. Feature selection step done here
    
    Args:
        row (pd.DataFrame): Path to the CSV file.
        dbn (int): Date block number used to filter columns.

    Returns:
        list: List of column names to read.
    """
    columns = row.columns

    cols = []
    for col in columns:
        l = col.split('$')
        if len(l) == 1:
            cols.append(col)
            continue

        name = l[0]
        num=int(l[1])
        dbn_diff = dbn - num
        
        if 'value_shop_id_item_id' in col and np.isclose(dbn_diff,0):#target
            cols.append(col)

        if dbn_diff<=0:
            continue

        if 'ema' in name and dbn_diff <= 2:
            cols.append(col)
            continue
        elif 'value_shop_id_item_id' in name and (dbn_diff <= 3 or dbn_diff == 6 or dbn_diff == 12):
            cols.append(col)
            continue
        elif 'value_price' in name and dbn_diff <= 1:
            cols.append(col)
            continue
        elif 'value' in name and dbn_diff <= 3:
            cols.append(col)
            continue
        elif 'diff' in name and dbn_diff == 1:
            cols.append(col)
            continue
        elif 'change' in name and dbn_diff <= 2:
            cols.append(col)
            continue


    return cols


def prepare_data_validation_boosting(row, dbn):#THIS FUNCTION MUUUUUST MATCH THE ONE IN utils.py
    
    
    lag_cols = []
    for col in row.columns:
        
            
        splitted = col.split('$')
        if len(splitted) == 1:
                lag_cols.append(col)
                continue
        for db in range(1,dbn):
            
            if db == int(splitted[1]):
                
                lag_cols.append(col)

    X = row[lag_cols]
    
    return X

def make_X_lag_format(data, dbn):#same as in fsp
    """
    Converts columns with date block numbers to a lag format for specified date block number.

    Args:
        data (pd.DataFrame): Data containing columns with date block numbers. Columns must have format name$dbn
        dbn (int): Current date block number for calculating lags.

    Returns:
        pd.DataFrame: Data with lagged columns for the specified date block number. Columns will have format name_lag;{lag number}
    """
    
    lag_cols = defaultdict()
    for col in data.columns:
        splitted = col.split('$')
        if len(splitted) == 1:
            continue
        
        lag_cols[col] = splitted[0] + '_lag;' + str(dbn - int(splitted[1]))

    #print(lag_cols)
    data = data.rename(columns=dict(lag_cols))
    #print(data.columns)
    return data

def append_some_columns(X_train, dbn):#same as in fsp
    """
    Adds additional columns like date_block_num and month to the training data.

    Args:
        X_train (pd.DataFrame): Training data.
        dbn (int): Current date block number.

    Returns:
        pd.DataFrame: Training data with additional columns.
    """
    X_train['date_block_num'] = dbn
    X_train['month'] = dbn%12
    return X_train