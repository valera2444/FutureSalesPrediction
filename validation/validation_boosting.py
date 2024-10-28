import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import root_mean_squared_error
from collections import defaultdict


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

import time

#np.random.seed(42)

SOURCE_PATH = 'data/merged.csv'
def prepare_past_ID_s(data_train):
    data_train['shop_item'] = [tuple([shop, item]) for shop, item in zip(data_train['shop_id'], data_train['item_id'])]
    #34 block contains A LOT more shop_item than others
    shop_item_pairs_in_dbn = data_train.groupby('date_block_num')['shop_item'].apply(np.unique)
    data_train = data_train.drop(['shop_item'], axis=1)
    
    shop_item_pairs_WITH_PREV_in_dbn = shop_item_pairs_in_dbn.copy()
    
    print(np.array(shop_item_pairs_WITH_PREV_in_dbn.index))
    arr = np.array(shop_item_pairs_WITH_PREV_in_dbn.index)
    
    for block in arr[arr>=0]:
        if block == 0:
            continue

        
        arr = np.append(shop_item_pairs_WITH_PREV_in_dbn[block -1],
                                                            shop_item_pairs_in_dbn[block-1])
        
        
        shop_item_pairs_WITH_PREV_in_dbn[block] = np.unique(np.append(shop_item_pairs_WITH_PREV_in_dbn[block -1],
                                                            shop_item_pairs_in_dbn[block-1]))
        print(len(shop_item_pairs_WITH_PREV_in_dbn[block]))

    return shop_item_pairs_in_dbn, shop_item_pairs_WITH_PREV_in_dbn

def prepare_past_ID_s_CARTESIAN(data_train):
    data_train['shop_item'] = [tuple([shop, item]) for shop, item in zip(data_train['shop_id'], data_train['item_id'])]
    #34 block contains A LOT more shop_item than others
    shop_item_pairs_in_dbn = data_train.groupby('date_block_num')['shop_item'].apply(np.unique)
    data_train = data_train.drop(['shop_item'], axis=1)
    
    shop_item_pairs_WITH_PREV_in_dbn = np.array([None] * len(shop_item_pairs_in_dbn))
    
    #print(np.array(shop_item_pairs_WITH_PREV_in_dbn.index))
    

    cartesians = []
    for dbn in shop_item_pairs_in_dbn.index:
        val = shop_item_pairs_in_dbn[dbn]

        shops = np.unique(list(zip(*val))[0])
        items = np.unique(list(zip(*val))[1])
    
        cartesian_product = np.random.permutation (np.array(np.meshgrid(shops, items)).T.reshape(-1, 2))
        #print(cartesian_product)
        cartesians.append(cartesian_product)
        
    
    shop_item_pairs_WITH_PREV_in_dbn[0] = cartesians[0]
    
    for block in shop_item_pairs_in_dbn.index:
        if block == 0:
            continue
        arr = np.append(shop_item_pairs_WITH_PREV_in_dbn[block - 1],
                             cartesians[block - 1], axis=0)#shop_item_pairs_WITH_PREV_in_dbn doesnt contain 34 month
        
        shop_item_pairs_WITH_PREV_in_dbn[block] = np.unique(arr, axis=0)
        print(len(shop_item_pairs_WITH_PREV_in_dbn[block]))
    return shop_item_pairs_in_dbn, shop_item_pairs_WITH_PREV_in_dbn

def make_X_lag_format(data, dbn):
    """
    transform X to lag format
    columns with dbn in names become lag_0, dbn-1 - lag_1 etc.
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

def prepare_train(data, valid ):
    """
    returns one batch of merged data with required IDs from valid
    """
    #print(data)
    valid_shop_item = valid
    valid_shop_item = list(zip(*valid_shop_item))
    df = pd.DataFrame({'item_id':valid_shop_item[1],'shop_id':valid_shop_item[0]} )
    data = df.merge(data, on=['shop_id','item_id'], how='inner').fillna(0)

    return data


def prepare_val(data, valid ):
    """
    returns one batch of merged data with required IDs from valid
    """
    
    df = pd.DataFrame({'item_id':valid[:,1],'shop_id':valid[:,0]} )
    data = df.merge(data, on=['shop_id','item_id'], how='inner').fillna(0)
    #print('prepare_val, data:',len(data))
    return data

def prepare_data_train_boosting(data, valid, dbn):
    """
    
    """
    train = prepare_train (data, valid)
    lag_cols = []
    for col in data.columns:
        
        splitted = col.split('$')
        if len(splitted) == 1:
                lag_cols.append(col)
                continue
        #if 'shop_item_cnt' not in col:
        #    continue
            
        for db in range(0,dbn-1):
            
            if db == int(splitted[1]):
                #print(col)
                lag_cols.append(col)

    #print(lag_cols)
    X = train[lag_cols]
    Y = train[f'value_shop_id_item_id${dbn-1}']
    
    return X, Y


def prepare_data_validation_boosting(data, valid, dbn):
    """
    
    """
    test = prepare_val (data, valid)
    
    lag_cols = []
    for col in test.columns:
        
            
        splitted = col.split('$')
        if len(splitted) == 1:
                lag_cols.append(col)
                continue
        #if 'shop_item_cnt' not in col:
        #    continue
        for db in range(1,dbn):
            
            if db == int(splitted[1]):
                #print(db, int(''.join(re.findall(r'\d+', col))))
                lag_cols.append(col)

    X = test[lag_cols]
    Y = test[f'value_shop_id_item_id${dbn}']#value_shop_id_item_id
    
    return X, Y

def select_columns_for_reading(path, dbn):
   
    columns = pd.read_csv(path, nrows=0).columns.tolist()

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


        if 'ema' in name and dbn_diff <= 3:
            cols.append(col)
        elif 'value_shop_id_item_id' in name and (dbn_diff <= 6 or dbn_diff == 12):
            cols.append(col)
        elif 'value_price' in name and dbn_diff <= 1:
            cols.append(col)
        elif 'value' in name and dbn_diff <= 3:
            cols.append(col)
        elif 'diff' in name and dbn_diff == 1:
            cols.append(col)
        elif 'change' in name and dbn_diff <= 2:
            cols.append(col)


    return cols

def create_batch_train(batch_size, dbn, shop_item_pairs_WITH_PREV_in_dbn, batch_size_to_read):
    """
    
    """
    
    train = np.random.permutation (shop_item_pairs_WITH_PREV_in_dbn[dbn])#-1?????????

    #chunk_num =  len(train)// batch_size if len(train)%batch_size==0  else   len(train) // batch_size + 1#MAY BE NEED TO CORRECT
    chunk_num =  len(train)// batch_size if len(train)>=batch_size else 1#MAY BE NEED TO CORRECT
    columns = select_columns_for_reading(SOURCE_PATH, dbn-1)#-1?????
    for idx in range(chunk_num):#split shop_item_pairs_WITH_PREV_in_dbn into chuncks
        t1 = time.time()
        l_x=[]
        l_y=[]
        merged = pd.read_csv('data/merged.csv', chunksize=batch_size_to_read, skipinitialspace=True, usecols=columns)
        l_sum = 0
        for chunck in merged:#split merged into chuncks
            
            l =  prepare_data_train_boosting(chunck,train[idx*batch_size:(idx+1)*batch_size], dbn) 
            #print(len(l[0]))
            l_sum += len(l[0])
            l_x.append( l[0] )
            l_y. append(l[1])
        
        if len(l_x) == 0:
            yield [None, None]
        print('create_batch_train, 203:',l_sum)
        l_x = pd.concat(l_x)
        l_y = pd.concat(l_y)

        t2 = time.time()
        print('batch creation time [create_batch_train, 212],',t2-t1)
        yield [l_x, l_y]#, test

def create_batch_val(batch_size, dbn, shop_item_pairs_in_dbn, batch_size_to_read):
    """
    
    """
    
    val = shop_item_pairs_in_dbn[dbn]
    
    shops = np.unique(list(zip(*val))[0])
    items = np.unique(list(zip(*val))[1])

    cartesian_product = np.random.permutation (np.array(np.meshgrid(shops, items)).T.reshape(-1, 2))
    
    chunk_num =  len(cartesian_product)// batch_size if len(cartesian_product)%batch_size==0  else   len(cartesian_product) // batch_size + 1#MAY BE NEED TO CORRECT

    columns = select_columns_for_reading(SOURCE_PATH, dbn)


    for idx in range(chunk_num):
        merged = pd.read_csv('data/merged.csv', chunksize=batch_size_to_read, skipinitialspace=True, usecols=columns)
        l_x=[]
        l_y=[]
        l_sum=0
        cartesian = cartesian_product[idx*batch_size:(idx+1)*batch_size]

        for chunck in merged:
            #print('chunck',len(chunck))
            #print('cartesian_product',len(cartesian))
            
            l =  prepare_data_validation_boosting(chunck,cartesian, dbn) 
            l_sum+=len(l[0])
            l_x.append( l[0] )
            l_y. append( l[1] )

        if len(l_x) == 0:
            yield [None, None]
        print('create_batch_val,243:',l_sum)
        l_x = pd.concat(l_x)
        l_y = pd.concat(l_y)


        yield [l_x,l_y]#, test


def select_columns(X_train, dbn):#WHEN LINEAR MODELS, X_train = append_some_columns(X_train,dbn) - to comment
    X_train = append_some_columns(X_train,dbn)
    cols=[]
    for col in X_train.columns:
        l = col.split(';')
        if len(l) == 1:
            cols.append(col)
            continue
        name = l[0]
        num = int(l[1])
        if 'ema' in name:
           if num <= 3:
                cols.append(col)
                continue
        if 'value_shop_id_item_id' in name:
            if num <=6 or num == 12:
                cols.append(col)
                continue
        if 'value_shop_id_lag' in name:
            continue

        if 'value_price' in name:
            if num <= 1:
                cols.append(col)
                continue
            
        if 'value' in name:
            if num <=3:
                cols.append(col)
                continue
            
        if 'diff' in name:
            if num == 1:
                cols.append(col)
                continue
            continue
            
        if 'change' in name:

            if num <= 2:
                cols.append(col)
                continue

            continue
        
    return X_train[cols]

def append_some_columns(X_train, dbn):
    X_train['date_block_num'] = dbn
    X_train['month'] = dbn%12
    return X_train

def train_model(model, batch_size, val_month, shop_item_pairs_WITH_PREV_in_dbn,batch_size_to_read,epochs):
    
    first=True
    rmse = 0
    c=0
    columns_order=None
    
    Y_true_l = []
    preds_l = []
    for epoch in range(epochs):
        for X_train,Y_train  in create_batch_train(batch_size, val_month,shop_item_pairs_WITH_PREV_in_dbn,batch_size_to_read):
            #print(f'train on batch {c} started')
            t1_batch = time.time()
            t1_data_prep = time.time()
            #print(f'data preparation on batch {c} started')
            if X_train is None:
                print('None')
                continue

            if type(model) in [Lasso,SVC]:
                #print(X_train.columns)
                X_train.drop('shop_id', inplace=True, axis=1) 
                X_train.drop('item_category_id', inplace=True, axis=1) 
                X_train.drop('item_id', inplace=True, axis=1)
                X_train.drop('city', inplace=True, axis=1)
                X_train.drop('shop_id', inplace=True, axis=1)
            else:
                #print(list(X_train.columns))
            
                #X_train = X_train.drop('item_id', axis=1)
                #X_train['shop_id'] = X_train['shop_id'].astype('category')
                #X_train['item_category_id'] = X_train['item_category_id'].astype('category')
                #X_train['city'] = X_train['city'].astype('category')
                #X_train['super_category'] = X_train['super_category'].astype('category')
                
                pass
            
            
                
            Y_train = np.clip(Y_train,0,20)
            
            if X_train.empty:
                print('None')
                continue
            
            X_train = make_X_lag_format(X_train, val_month-1)
            
            #X_train=select_columns(X_train, val_month-1)
            
                
            columns_order=X_train.columns

            t2_data_prep = time.time()
            #print(f'data preparation on batch {c} time:',t2_data_prep-t1_data_prep)
            #print('model fitting started')
            t1_fit = time.time()
            if c == 0:
                pass
                #print('train columns')
                #print(X_train.columns)
            if type(model) in [Lasso,SVC]:
                model.fit(X_train, Y_train)
                y_train_pred = model.predict(X_train)
            
            elif type(model) == LGBMRegressor:
                if first:
                    model.fit(X_train, Y_train)
                    first=False
                else:
                    model.fit(X_train, Y_train, init_model=model)
                y_train_pred = model.predict(X_train, validate_features=True)

            elif type(model) == xgb.XGBRegressor:
                if first:
                    model=model.fit(X_train, Y_train)
                    first=False
                else:
                    print(model.get_booster())
                    #Works not as expected
                    model=model.fit(X_train, Y_train, xgb_model=model.get_booster())
                    
                    
                y_train_pred = model.predict(X_train)  

            elif type(model) == RandomForestRegressor:
                model.fit(X_train, Y_train)
                y_train_pred = model.predict(X_train)  
            t2_fit = time.time()
            #print(f'model fitting time on batch {c},',t2_fit - t1_fit)
            
            Y_true_l.append(Y_train)
            preds_l.append(y_train_pred)
            t2_batch = time.time()
            print(f'train on batch {c} time,',t2_batch-t1_batch)
            c+=1
        
    train_rmse = root_mean_squared_error(pd.concat(Y_true_l), np.concat(preds_l))
    print('train_rmse, ',train_rmse)
           

    return model, columns_order

def validate_model(model,batch_size, val_month, columns_order, shop_item_pairs_in_dbn, batch_size_to_read):
    rmse = 0
    c=0
    
    val_preds = []
    Y_true_l = []
    preds_l = []
    #create_batch_train(merged,batch_size, val_month) - return train set, where Y_val
    #is shop_item_cnt_month{val_month}
    for X_val, Y_val in create_batch_val(batch_size, val_month, shop_item_pairs_in_dbn, batch_size_to_read):#but then cartesian product used
        if X_val is None:
                    continue
        
        if type(model) in [sklearn.linear_model._coordinate_descent.Lasso,
                          SVC]:
            
            X_val.drop('shop_id', inplace=True, axis=1) 
            X_val.drop('item_category_id', inplace=True, axis=1) 
            X_val.drop('item_id', inplace=True, axis=1)
            X_val.drop('city', inplace=True, axis=1)
            X_val.drop('shop_id', inplace=True, axis=1)
            

        else:
            
            #X_val = X_val.drop('item_id', axis=1)
            #X_val['shop_id'] = X_val['shop_id'].astype('category')
            #X_val['item_category_id'] = X_val['item_category_id'].astype('category')
            #X_val['city'] = X_val['city'].astype('category')
            #X_val['super_category'] = X_val['super_category'].astype('category')
                    
            pass
            
        
            
        Y_val = np.clip(Y_val,0,20)
        
        
        X_val = make_X_lag_format(X_val, val_month)
        
        #X_val=select_columns(X_val, val_month)
        X_val = X_val[columns_order]

        if type(model) in [Lasso,SVC]:
            y_val_pred = model.predict(X_val)#lgb - validate features
            
        elif type(model) ==LGBMRegressor:
            y_val_pred = model.predict(X_val, validate_features=True)#lgb - validate features

        elif type(model) == xgb.XGBRegressor:
            y_val_pred = model.predict(X_val)   

        elif type(model) == RandomForestRegressor:
            y_val_pred = model.predict(X_val)  

        y_val_pred = np.clip(y_val_pred,0,20)

        
        
        
        preds_l.append(y_val_pred)
        Y_true_l.append(Y_val)
        
        c+=1
        
        val_preds.append(y_val_pred)

    
    val_rmse = root_mean_squared_error(pd.concat(Y_true_l), np.concat(preds_l))
    print('val rmse, ',val_rmse)

    return val_preds, val_rmse

def validate_ML(model,batch_size,start_val_month, shop_item_pairs_in_dbn, shop_item_pairs_WITH_PREV_in_dbn,batch_size_to_read,epochs):
    """
    Function for validating model
    
    """
    
    val_errors = []
    
    val_preds=[]
    
    
    for val_month in range(start_val_month, 34):

        print(f'month {val_month} started')
        t1 = time.time()
        
        print('month', val_month%12)

        model,columns_order = train_model(model, batch_size, val_month, shop_item_pairs_WITH_PREV_in_dbn,batch_size_to_read,epochs)

        t2=time.time()
        print(f'model training on {val_month} time,',t2-t1)
        print('feature importances, ')
        print(list(model.feature_names_in_[np.argsort( model.feature_importances_)][::-1]))
        
        #dump_list = model.get_booster().get_dump()
        #num_trees = len(dump_list)
        
        #print('n_estimators:', model.n_estimators_)
        t1 = time.time()
        #print(f'validation on month {val_month} started')
        val_pred, val_error = validate_model(model,batch_size, val_month,columns_order, shop_item_pairs_in_dbn,batch_size_to_read)
        t2 = time.time()
        print(f'validation time on month {val_month},',t2-t1)
        val_errors.append(val_error)
        val_preds.append(val_pred)
        

    return val_errors, val_preds

def create_submission(model,batch_size, columns_order, shop_item_pairs_in_dbn,batch_size_to_read):
    val_month = 34
    test = pd.read_csv('../data_cleaned/test.csv')
    
    data_test = test
    PREDICTION = pd.DataFrame(columns=['shop_id','item_id','item_cnt_month'])
    Y_true_l=[]
    for X_val, Y_val in create_batch_val(batch_size, val_month, shop_item_pairs_in_dbn,batch_size_to_read):
        if type(model) in [sklearn.linear_model._coordinate_descent.Lasso,
                          SVC]:
            
            X_val.drop('shop_id', inplace=True, axis=1) 
            X_val.drop('item_category_id', inplace=True, axis=1) 
            X_val.drop('item_id', inplace=True, axis=1) 
            

        else:
            
            #X_val = X_val.drop('item_id', axis=1)
            #X_val['shop_id'] = X_val['shop_id'].astype('category')
            #X_val['item_category_id'] = X_val['item_category_id'].astype('category')
            #X_val['city'] = X_val['city'].astype('category')
            #X_val['super_category'] = X_val['super_category'].astype('category')
                    
            pass

        
        if X_val is None:
            continue
            
        Y_val = np.clip(Y_val,0,20)
        
        if X_val.empty:
            print('None')
            continue
            
        
        X_val = make_X_lag_format(X_val, val_month)
        #X_val=select_columns(X_val, val_month)
        X_val = X_val[columns_order]

        
        y_val_pred=model.predict(X_val)
        y_val_pred = np.clip(y_val_pred,0,20)#lgb - validate features
        Y_true_l.append(Y_val)
        
        
        app = pd.DataFrame({'item_id':X_val.item_id,'shop_id': X_val.shop_id, 'item_cnt_month':y_val_pred})
        PREDICTION = pd.concat([PREDICTION, app],ignore_index=True)

    #val_rmse = root_mean_squared_error(PREDICTION['item_cnt_month'], np.concat(Y_true_l))
    #print('val rmse, ',val_rmse)
    
    data_test = data_test.merge(PREDICTION,on=['shop_id','item_id'])[['ID','item_cnt_month']]
    return data_test

def create_submission_pipeline(merged, model,batch_size,shop_item_pairs_in_dbn, shop_item_pairs_WITH_PREV_in_dbn,batch_size_to_read,epochs):
    val_errors = []
    
    val_errors=[]

    #print(f'model training on 34 started')
    t1 = time.time()
    model,columns_order = train_model(model,batch_size, 34, shop_item_pairs_WITH_PREV_in_dbn,batch_size_to_read,epochs)
    t2 = time.time()
    print('training model time,',t2-t1)
    print('Feature importnaces in lgb:')
    
    print(model.feature_names_in_[np.argsort(model.feature_importances_)][::-1])
    #print('n_estimators:', model.n_estimators_)
    #print('submission creation started')
    t1 = time.time()
    data_test = create_submission(model,batch_size,columns_order, shop_item_pairs_in_dbn,batch_size_to_read)
    t2 = time.time()
    print('submission creation time,', t2-t1)

    return data_test
    

if __name__ == '__main__':

    data_train = pd.read_csv('../data_cleaned/data_train.csv')
    test = pd.read_csv('../data_cleaned/test.csv')
    test['date_block_num'] = 34
    data_train = pd.concat([data_train,test ], ignore_index=True).drop('ID', axis=1).fillna(0)


    shop_item_pairs_in_dbn, shop_item_pairs_WITH_PREV_in_dbn = prepare_past_ID_s_CARTESIAN(data_train)


    
    
    start_val_month=22
    #model = LGBMRegressor(verbose=-1,n_jobs=8, num_leaves=750, n_estimators = 1700,  learning_rate=0.001)
    #model =RandomForestRegressor(max_depth = 10, n_estimators = 100,n_jobs=8)
    model = xgb.XGBRegressor(eta=0.001, max_leaves=640,nthread=8,device='gpu', enable_categorical=True,n_estimators=5000)
   
    batch_size=70000000
    batch_size_to_read=200000000

    is_create_submission=True
    epochs=1

    if is_create_submission:
        print('submission creation started...')
        submission = create_submission_pipeline(merged=None, 
                                            model=model,
                                            batch_size=batch_size,
                                            shop_item_pairs_in_dbn=shop_item_pairs_in_dbn,
                                            shop_item_pairs_WITH_PREV_in_dbn=shop_item_pairs_WITH_PREV_in_dbn,
                                            batch_size_to_read=batch_size_to_read,
                                            epochs=epochs
                                            )
        submission.to_csv('submission.csv', index=False)
        print(submission.describe())

    else:
        print('validation started...')
        val_errors, val_preds = validate_ML(
                                            model=model,
                                            batch_size=batch_size,
                                            start_val_month=start_val_month, 
                                            shop_item_pairs_in_dbn=shop_item_pairs_in_dbn,
                                            shop_item_pairs_WITH_PREV_in_dbn=shop_item_pairs_WITH_PREV_in_dbn,
                                            batch_size_to_read=batch_size_to_read,
                                            epochs=epochs
        )
        print(val_errors)