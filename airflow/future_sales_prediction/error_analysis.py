import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

import os
import mlflow

from gcloud_operations import upload_folder, upload_file, download_file, download_folder

mapping = {
    0: [[0, 12, 24], 'January'],
    1: [[1, 13, 25], 'February'],
    2: [[2, 14, 26], 'March'],
    3: [[3, 15, 27], 'April'],
    4: [[4, 16, 28], 'May'],
    5: [[5, 17, 29], 'June'],
    6: [[6, 18, 30], 'July'],
    7: [[7, 19, 31], 'August'],
    8: [[8, 20, 32], 'September'],
    9: [[9, 21, 33], 'October'],
    10: [[10, 22, 34], 'November'],
    11: [[11, 23, 35], 'December'],
}



def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--path_for_merged', type=str, help='folder where error arrays stored (same as folder where merged.csv stored)')
    parser.add_argument('--path_data_cleaned', type=str)
    parser.add_argument('--path_artifact_storage', type=str, help='folder name in s3 storage where artifacts will be stored')
    parser.add_argument('--test_month', type=int, help='month to create submission on')


    args = parser.parse_args()
    return args


args = read_args()
bucket_name = os.environ.get("BUCKET_NAME")

mlflow.set_tracking_uri(uri="http://mlflow:5000")
run_id = mlflow.search_runs(experiment_names=['create_submission'],filter_string=f"run_name='{args.run_name}'").iloc[0]['run_id']

def download_files(args):
    if not os.path.exists(args.path_data_cleaned):
        download_folder(bucket_name,args.path_data_cleaned)
    
    if not os.path.exists( f'{args.path_for_merged}/item_dbn_diff.csv'):
        download_file(bucket_name, f'{args.path_for_merged}/item_dbn_diff.csv')
    if not os.path.exists(f'{args.path_for_merged}/val_preds.npy'):
        download_file(bucket_name, f'{args.path_for_merged}/val_preds.npy')
    if not os.path.exists(f'{args.path_for_merged}/val_true.npy'):
        download_file(bucket_name, f'{args.path_for_merged}/val_true.npy')
    if not os.path.exists(f'{args.path_for_merged}/merged.csv'):
        download_file(bucket_name,f'{args.path_for_merged}/merged.csv')




def select_columns_for_reading(path, dbn):
   
    columns = pd.read_csv(path, nrows=0).columns.tolist()

    cols = []
    for col in columns:
        l = col.split('$')
        if len(l) == 1:
            cols.append(col)
            continue

    return cols


def parse_city(shop_name):

    if shop_name.split()[0] == '!Якутск':
        return  'Якутск'

    if shop_name.split()[0] == 'Сергиев':
            return  'Сергиев Посад'
    else:
        return shop_name.split()[0]


def diff(target, pred):
    return target-pred
def mse(target, pred):
    return (target-pred)**2

def mae(target, pred):
    return np.abs(target-pred)


def smape(A,F):
    return 100 * (2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def mape(A,F):
    return 100 * np.abs(F - A) / np.abs(A)




def prepare_df(args):
    SOURCE_PATH = f'{args.path_for_merged}/merged.csv'

    cols = select_columns_for_reading(SOURCE_PATH,0)
    merged = pd.read_csv(SOURCE_PATH, usecols=cols)
        
    dbn_diff = pd.read_csv(f'{args.path_for_merged}/item_dbn_diff.csv')
    dbn_diff_selected = dbn_diff[['shop_id','item_id',*[str(i) for i in range(args.test_month-12,args.test_month+1)]]]
    df_diff = pd.DataFrame({'shop_id':[], 'item_id':[],'date_block_num':[],'dbn_diff':[]})
    for month in range(args.test_month-12,args.test_month+1):
        shop_id = dbn_diff_selected.shop_id
        item_id = dbn_diff_selected.item_id
        dbn = month
        diffs = dbn_diff_selected[str(month)]
        to_app = pd.DataFrame({'shop_id':shop_id, 'item_id':item_id,'date_block_num':[dbn] * len(item_id),'dbn_diff':diffs})
        df_diff=pd.concat([df_diff,to_app])


    categories_f = pd.read_csv(f'{args.path_data_cleaned}/item_categories.csv')
    shops_f = pd.read_csv(f'{args.path_data_cleaned}/shops.csv')

    categories_f['super_category']=categories_f['item_category_name'].apply(lambda a: a.split()[0])
    shops_f['city']=shops_f['shop_name'].apply(parse_city)


    merged=merged.drop(['city'],axis=1)
    merged=merged.drop(['super_category'],axis=1)
    merged=merged.merge(shops_f, how='left').merge(categories_f, how='left')


    #Assume that monthes are [args.test_month - 12,args.test_month]  
    preds = np.load(f'{args.path_for_merged}/val_preds.npy',allow_pickle=True)#preds_l.append(y_val_pred)
    real = np.load(f'{args.path_for_merged}/val_true.npy',allow_pickle=True)#Y_true_l.append([np.array(Y_val).flatten(), np.array(shop_id).flatten(), np.array(item_id).flatten()])

    df = pd.DataFrame({'shop_id':[], 'item_id':[],'date_block_num':[],'sales':[],'preds':[]})
    for month in range(len(real)):
        month_num = month+args.test_month-12
        sales = real[month][:,0][0]#Critical
        shop_id=real[month][:,1][0]
        item_id=real[month][:,2][0]
        preds_val = preds[month][0]
        dbn = [month_num] * len(shop_id)
        to_app=pd.DataFrame({'shop_id':shop_id, 'item_id':item_id,'date_block_num':dbn,'sales':sales, 'preds':preds_val})
        df=pd.concat([df,to_app])

    df=df.merge(merged).merge(df_diff)
    return df


bucket_name = bucket_name




download_files(args)

df = prepare_df(args)


total_sails=df.groupby('date_block_num').agg({'sales':'mean','preds':'mean'})

plt.plot(np.arange(args.test_month-12,args.test_month,dtype=int),total_sails['sales'], label='Total sales target' )
plt.plot(np.arange(args.test_month-12,args.test_month,dtype=int),total_sails['preds'], label='Total sales predictions' )
plt.legend()

plt.xlabel('Sales')
plt.xlabel('date block num')

plt.savefig('sales_preds_per_dbn.png')
mlflow.log_artifact(local_path='sales_preds_per_dbn.png', artifact_path= f'{args.path_artifact_storage}/err_analysis', run_id=run_id) 
#s3c.upload_file('sales_preds_per_dbn.png', bucket_name, f'{args.path_artifact_storage}/err_analysis/sales_preds_per_dbn.png')



df['mae_errors'] = mae(df['preds'] , df['sales'] )
df['smape_errors'] = smape(df['preds'] , df['sales'] ).fillna(0)
df['diff'] = diff(df['preds'] , df['sales'] )

fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(20,7))
df_sel = df

sns.barplot(x=df_sel['dbn_diff'],y=df_sel['sales'],ax=ax[0])
ax[0].set_title('Mean sales per dbn_diff')
ax[0].tick_params(axis='x', labelrotation=90,labelsize=7)

sns.barplot(x=df_sel['dbn_diff'],y=df_sel['mae_errors'],ax=ax[1])
sns.barplot(x=df_sel['dbn_diff'],y=df_sel['smape_errors'],ax=ax[2])
ax[1].tick_params(axis='x', labelrotation=90,labelsize=7)
ax[1].set_title('MAE error for dbn_diff')
ax[2].tick_params(axis='x', labelrotation=90,labelsize=7)
ax[2].set_title('SMAPE error for dbn_diff');

fig.savefig('mean_sales_errors_per_dbn_diff.png')
mlflow.log_artifact(local_path='mean_sales_errors_per_dbn_diff.png', artifact_path= f'{args.path_artifact_storage}/err_analysis', run_id=run_id) 
#s3c.upload_file('mean_sales_errors_per_dbn_diff.png', bucket_name, f'{args.path_artifact_storage}/err_analysis/mean_sales_errors_per_dbn_diff.png')





fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15,7))
df_sel = df

sns.barplot(x=df_sel['date_block_num'],y=df_sel['sales'],ax=ax[0])
ax[0].set_title('Mean sales per date_block_num')
ax[0].tick_params(axis='x', labelrotation=90)

sns.barplot(x=df_sel['date_block_num'],y=df_sel['mae_errors'],ax=ax[1])
sns.barplot(x=df_sel['date_block_num'],y=df_sel['smape_errors'],ax=ax[2])
ax[1].tick_params(axis='x', labelrotation=90)
ax[1].set_title('MAE error for date_block_num')
ax[2].tick_params(axis='x', labelrotation=90)
ax[2].set_title('SMAPE error for date_block_num');

fig.savefig('mean_sales_errors_per_date_block_num.png')
mlflow.log_artifact(local_path='mean_sales_errors_per_date_block_num.png', artifact_path= f'{args.path_artifact_storage}/err_analysis', run_id=run_id) 
#s3c.upload_file('mean_sales_errors_per_date_block_num.png', bucket_name, f'{args.path_artifact_storage}/err_analysis/mean_sales_errors_per_date_block_num.png')






fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15,7))

df_sel = df
sns.barplot(x=df_sel['super_category'],y=df_sel['sales'],ax=ax[0])
ax[0].set_title('Mean sales per super_category')
ax[0].tick_params(axis='x', labelrotation=90)

sns.barplot(x=df_sel['super_category'],y=df_sel['mae_errors'],ax=ax[1])
sns.barplot(x=df_sel['super_category'],y=df_sel['smape_errors'],ax=ax[2])
ax[1].tick_params(axis='x', labelrotation=90)
ax[1].set_title('MAE error for super_category')
ax[2].tick_params(axis='x', labelrotation=90)
ax[2].set_title('SMAPE error for super_category');

fig.savefig('mean_sales_errors_per_super_category.png')
mlflow.log_artifact(local_path='mean_sales_errors_per_super_category.png', artifact_path= f'{args.path_artifact_storage}/err_analysis', run_id=run_id) 
#s3c.upload_file('mean_sales_errors_per_super_category.png', bucket_name, f'{args.path_artifact_storage}/err_analysis/mean_sales_errors_per_super_category.png')



fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15,7))
df_sel = df
sns.barplot(x=df_sel['city'],y=df_sel['sales'],ax=ax[0])
ax[0].set_title('Mean sales per city')
ax[0].tick_params(axis='x', labelrotation=90)

sns.barplot(x=df_sel['city'],y=df_sel['mae_errors'],ax=ax[1])
sns.barplot(x=df_sel['city'],y=df_sel['smape_errors'],ax=ax[2])
ax[1].tick_params(axis='x', labelrotation=90)
ax[1].set_title('MAE error for city')
ax[2].tick_params(axis='x', labelrotation=90)
ax[2].set_title('SMAPE error for city');

fig.savefig('mean_sales_errors_per_city.png')
mlflow.log_artifact(local_path='mean_sales_errors_per_city.png', artifact_path= f'{args.path_artifact_storage}/err_analysis', run_id=run_id) 
#s3c.upload_file('mean_sales_errors_per_city.png', bucket_name, f'{args.path_artifact_storage}/err_analysis/mean_sales_errors_per_city.png')






fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15,7))
df_sel = df
sns.histplot(x=df_sel['dbn_diff'],ax=ax[0])
sns.boxplot(x=df_sel['sales'],y=df_sel['mae_errors'],ax=ax[1])
sns.boxplot(x=df_sel['sales'],y=df_sel['smape_errors'],ax=ax[2])

ax[0].set_title('Hist of monthes since item have been on the market')

ax[1].tick_params(axis='x', labelrotation=90)
ax[1].set_title('MAE error for different sales')

ax[2].tick_params(axis='x', labelrotation=90)
ax[2].set_title('SMAPE error for different sales');
fig.savefig('hist_n_errors_per_dbn_diff.png')
mlflow.log_artifact(local_path='hist_n_errors_per_dbn_diff.png', artifact_path= f'{args.path_artifact_storage}/err_analysis', run_id=run_id) 
#s3c.upload_file('hist_n_errors_per_dbn_diff.png', bucket_name, f'{args.path_artifact_storage}/err_analysis/hist_n_errors_per_dbn_diff.png')
#SMAPE is undefined in 0





df_large_errors = df.query('diff > 5')
plt.hist(df_large_errors.sales,bins=50);
plt.xlabel('count')
plt.ylabel('sales')
plt.title('Histogram of sales with error > 5')
plt.savefig('histogram_of sales_with_error > 5.png')
mlflow.log_artifact(local_path='histogram_of sales_with_error > 5.png', artifact_path= f'{args.path_artifact_storage}/err_analysis', run_id=run_id) 
#s3c.upload_file('histogram_of sales_with_error > 5.png', bucket_name, f'{args.path_artifact_storage}/err_analysis/histogram_of sales_with_error > 5.png')




df_large_errors = df.query('diff < -5')
plt.hist(df_large_errors.sales,bins=50);
plt.xlabel('count')
plt.ylabel('sales')
plt.title('Histogram of sales with error < -5')
plt.savefig('histogram_of_sales_with_error < -5.png')
mlflow.log_artifact(local_path='histogram_of_sales_with_error < -5.png', artifact_path= f'{args.path_artifact_storage}/err_analysis', run_id=run_id) 
#s3c.upload_file('histogram_of_sales_with_error < -5.png', bucket_name, f'{args.path_artifact_storage}/err_analysis/histogram_of_sales_with_error < -5.png')



df_large_errors = df.query('mae_errors > 10')
plt.violinplot(df_large_errors['diff']);

plt.ylabel('diff')
plt.title('violinplot of sales with MAE > 10')
plt.savefig('violinplot of sales with MAE > 10.png')
mlflow.log_artifact(local_path='violinplot of sales with MAE > 10.png', artifact_path= f'{args.path_artifact_storage}/err_analysis', run_id=run_id) 
#s3c.upload_file('histogram_of_sales_with_error < -5.png', bucket_name, f'{args.path_artifact_storage}/err_analysis/histogram_of_sales_with_error < -5.png')




fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(15,7))
ax[0].hist(df.dbn_diff,bins=400)
ax[0].set_xlabel('Monthes since first sale')
ax[0].set_title('Distribution of date block num difference in source data')
ax[1].hist(df_large_errors.dbn_diff, bins=400);
ax[1].set_xlabel('Monthes since first sale')
ax[1].set_title('Distribution of date block num difference in data with MAE > 19')
plt.savefig('distributions_of_dbns.png')
mlflow.log_artifact(local_path='distributions_of_dbns.png', artifact_path= f'{args.path_artifact_storage}/err_analysis', run_id=run_id) 
#s3c.upload_file('distributions_of_dbns.png', bucket_name, f'{args.path_artifact_storage}/err_analysis/distributions_of_dbns.png')




df_selected = df.query('item_id== 20949')
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(15,10))

df_selected_shop1=df_selected[df_selected['shop_name']=='Москва ТРК \"Атриум\"']
sns.lineplot(df_selected_shop1[['date_block_num','sales']], x='date_block_num',y='sales',label='sales', ax=ax[0,0])
sns.lineplot(df_selected_shop1[['date_block_num','preds']], x='date_block_num',y='preds',label='preds',ax=ax[0,0])
ax[0,0].set_title('Москва ТРК \"Атриум\"')

df_selected_shop2=df_selected[df_selected['shop_name']=='Москва ТЦ \"Новый век\" (Новокосино)']
sns.lineplot(df_selected_shop2[['date_block_num','sales']], x='date_block_num',y='sales',label='sales', ax=ax[0,1])
sns.lineplot(df_selected_shop2[['date_block_num','preds']], x='date_block_num',y='preds',label='preds',ax=ax[0,1])
ax[0,1].set_title('Москва ТЦ \"Новый век\" (Новокосино)')

df_selected_shop3=df_selected[df_selected['shop_name']=="Уфа ТК \"Центральный\""]
sns.lineplot(df_selected_shop3[['date_block_num','sales']], x='date_block_num',y='sales',label='sales', ax=ax[1,0])
sns.lineplot(df_selected_shop3[['date_block_num','preds']], x='date_block_num',y='preds',label='preds',ax=ax[1,0])
ax[1,0].set_title("Уфа ТК \"Центральный\"")

df_selected_shop4=df_selected[df_selected['shop_name']=="РостовНаДону ТЦ \"Мега\""]
sns.lineplot(df_selected_shop4[['date_block_num','sales']], x='date_block_num',y='sales',label='sales', ax=ax[1,1])
sns.lineplot(df_selected_shop4[['date_block_num','preds']], x='date_block_num',y='preds',label='preds',ax=ax[1,1])
ax[1,1].set_title("РостовНаДону ТЦ \"Мега\"")

fig.savefig('large_errors_far_after_entering_market')
mlflow.log_artifact(local_path='large_errors_far_after_entering_market.png', artifact_path= f'{args.path_artifact_storage}/err_analysis', run_id=run_id) 
#s3c.upload_file('large_errors_far_after_entering_market.png', bucket_name, f'{args.path_artifact_storage}/err_analysis/large_errors_far_after_entering_market.png')