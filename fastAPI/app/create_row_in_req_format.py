import pandas as pd
import itertools
import os
import boto3
import argparse
import numpy as np

def match(merged, row,new_row,  test_month, check):
    """Collects statistics per one batch for newcoming item based on collected data while preprocessing

    Args:
        merged (pd.Dataframe): final csv file
        row (pd.Dataframe): row with shop-item specific olumns (item, shop, category, super_category, city)
        new_row (pd.Dataframe): iteratively fiiled raw
        test_month ():
        check (dict): item and shop as keys and shop-item specific columns (item_id, shop_id, category, super_category, city) as values
    Returns:
        pd.Dataframe: dataframe with 1 row with collected statistics
    """
    
    def select_columns_by_exact_name(col_name, check, columns):
        result=[]
        if col_name in check['item']:
            for col in columns:
                if (col_name in col) and ('shop_id' not in col and 'city' not in col):
                    result.append(col)
        else:
            for col in columns:
                if (col_name in col) and ('item_id' not in col and 'item_category_id' not in col and 'super_category' not in col):
                    result.append(col)

        return result

    def select_columns_by_pair(pair, columns):
        """
        Selects columns that exactly match pair (eg columns with only such substring)) of strings
        Args:
            pair (tuple[str, str]): _description_
            columns (list[str]): _description_

        Returns:
            list[str]: list of matched columns
        """
        result=[]
        for col in columns:
                if (pair[0] in col and pair[1] in col):
                    result.append(col)

        return result

    
    
    #print(row)
    columns = merged.columns.tolist()
    
    for c in check['item'] + check['shop']:
        cols_single=[]
        common_rows = row.merge(merged, on=c, how='inner')
        if not common_rows.empty:
            col = select_columns_by_exact_name(c, check, columns)
            
            cols_single.append(col)
            new_row[col] = common_rows.iloc[0][col]

    
    for c in list(itertools.product(*[check['item'], check['shop']])):
        cols_double=[]
        common_rows = row.merge(merged, on=c, how='inner')
        
        if not common_rows.empty:
            if 'shop_id' in c and 'item_id' in c:
                print(common_rows)
            col = select_columns_by_pair(c, columns)
            cols_double.append(col)
            new_row[col] = common_rows.iloc[0][col]

    #Handle features where there is only iyem in name, not item_id
    common_rows = row.merge(merged, on='item_id', how='inner')
    if not common_rows.empty:
        cols = [col for col in columns if 'diff' in col or 'change' in col]
        new_row[cols] = common_rows.iloc[0][cols]

    #Handle item dbn diff (should be -1 if not seen this month or previously)
    if new_row['item_dbn_diff$1'].isna().any():
        new_row[[col for col in columns if 'item_dbn_diff' in col]] = -1
        new_row[f'item_dbn_diff${test_month}'] = 0

   
    return new_row.fillna(0)

def download_s3_folder(s3c, bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory to local_dir (creates if not exist)
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3c.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)
        #print(obj.key, target)

   
def parse_city(shop_name):
    """
    Extracts city name from shop name string.
    
    Args:
        shop_name (str): Name of the shop.

    Returns:
        str: Extracted city name.
    """

    if shop_name.split()[0] == '!Якутск':
        return  'Якутск'

    if shop_name.split()[0] == 'Сергиев':
            return  'Сергиев Посад'
    else:
        return shop_name.split()[0]

def create_big_5(partial_merged, shop_id, item_id):
    """Merges shop_id, item_id with theirs gloabal features( super_category, city, ctgeory)

    Args:
        partial_merged (pd.DataFrame): merged.csv only with required columns
        shop_id (int): _description_
        item_id (int): _description_

    Returns:
        pd.DataFrame: dataframe with shop, item, vategory, super_category, city
    """
    
    shop_city_pairs=pd.DataFrame({'shop_id':[shop_id],'item_id':[item_id]})
    
    shop_city = partial_merged.groupby('shop_id')['city'].unique().reset_index()
    shop_city_pairs=shop_city_pairs.merge(shop_city, how='left')
    shop_city_pairs['city'] = shop_city_pairs['city'].apply(lambda a:a[0])

    item_category = partial_merged.groupby('item_id')['item_category_id'].unique().reset_index()
    shop_city_pairs=shop_city_pairs.merge(item_category, how='left')
    shop_city_pairs['item_category_id'] = shop_city_pairs['item_category_id'].apply(lambda a:a[0])

    category_super_category = partial_merged.groupby('item_category_id')['super_category'].unique().reset_index()
    shop_city_pairs=shop_city_pairs.merge(category_super_category, how='left')
    shop_city_pairs['super_category'] = shop_city_pairs['super_category'].apply(lambda a:a[0])
    
    return shop_city_pairs[['shop_id','item_id','item_category_id','city','super_category']]

def create_5_plus_match(args):
    """
    Runs pipeline for row preaparation in format as merged.csv

    Args:
        args (object): Object of custom class with required fields

    Returns:
        pd.DataFrame: row as in merged.csv
    """
    shop_id = args.shop_id
    item_id = args.item_id

    partial_merged = pd.read_csv(f'{args.path_for_merged}/merged.csv', usecols=['city','shop_id','item_id','super_category','item_category_id'])

    if shop_id not in partial_merged.shop_id.values:
        print('No such shop')
        exit(-1)
        
    if item_id not in partial_merged.item_id.values:
        print('No such item')
        exit(-1)

    row = create_big_5(partial_merged, shop_id, item_id)
    
    
    merged = pd.read_csv(f'{args.path_for_merged}/merged.csv', chunksize=args.batch_size)
    check = {'item':['item_id','item_category_id','super_category'],'shop':['shop_id','city']}
    c=0

    columns=pd.read_csv(f'{args.path_for_merged}/merged.csv', nrows=0).columns.tolist()

    new_row = pd.DataFrame(columns=columns)
    new_row[check['item'] + check['shop']] = row[check['item'] + check['shop']]

    for merged_part in merged:
        new_row=match(merged_part, row, new_row, args.test_month,check)
        c+=1
        print(f"batch {c} searched")

    #print(new_row.isna())
    return new_row
