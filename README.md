# FutureSalesPrediction

## Brief explanation of validation_boosting.py datat preparation process

**validation_boosting.py** uses data from **data/merged.csv** . 
### Structure of merged.csv
The file `merged.csv` contains:
- **Monthly Features**: Columns related to the month-specific characteristics of each (shop, item) pair. These features are tagged with a date suffix in the format `{feature_name}$dbn`, where `dbn` is the `date_block_num` (e.g., `ema_item_category_id$10`).
- **Static Features**: Columns that are constant over time, such as static attributes of shops or items.


> **Note**: The file may contain fewer than 34 columns per monthy feature, as some features may not be available for all months from 0 to 33.

### `validation_boosting.py`: Data Preparation for Training and Validation
The script `validation_boosting.py` handles data preprocessing for training and validation sets.

#### Key Functions

1. **`prepare_past_ID_s_CARTESIAN`**
   - Creates two arrays: 
     - `shop_item_pairs_in_dbn`: Stores the Cartesian product of shops and items for each month. Each row `i` represents unique (shop, item) pairs for month `i`.
     - `shop_item_pairs_WITH_PREV_in_dbn`: Similar to `shop_item_pairs_in_dbn` but contains pairs for months ranging from 0 to `(i-1)`, providing historical data for each (shop, item) combination.

2. **`select_columns_for_reading`**
   - This function enables selective loading of specific columns from `merged.csv`, avoiding loading the entire dataset into memory, which optimizes resource usage.

3. **`make_X_lag_format`**
   - Transforms feature names to lagged format for modeling. For instance, if the model is validated on month 25 and needs the feature `ema_item_id$23`, the feature name is converted to `ema_item_id_lag_1` for training and `ema_item_id_lag_2` for validation. This allows consistent alignment of past features as "lag" variables for any target month.

### Train and Validation Set Creation
To validate the model on a specific month `k`:
- Use `(shop, item)` pairs from `shop_item_pairs_in_dbn[k]` to create validation set.
- Use `(shop, item)` pairs from `shop_item_pairs_WITH_PREV_in_dbn[k]` to create training set, providing the model with prior months’ data up to `k-1`.

## Usage and Execution
1. **Run Data Preparation**: Execute `prepara_data.py` to prepare the initial merged dataset.
2. **Train and Validate the Model**: Use `validation_boosting.py` with the necessary configurations to handle selective data loading, lag formatting, and partitioning into train and validation sets based on month.
