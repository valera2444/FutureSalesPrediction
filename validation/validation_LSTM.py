import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error

from collections import defaultdict

from torch.nn.functional import normalize
import torch
from torch.optim.lr_scheduler import StepLR

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
        shop_item_pairs_WITH_PREV_in_dbn[block] = np.unique(np.append(shop_item_pairs_WITH_PREV_in_dbn[block -1],
                                                                      #shop_item_pairs_WITH_PREV_in_dbn[block]))
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
                             cartesians[block - 1], axis=0)
        
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
        
        
        lag_cols[col] = splitted[0] + '_lag_' + str(dbn - int(splitted[1]))

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
    data = df.merge(data, on=['shop_id','item_id'], how='left').fillna(0)
    
    return data


def prepare_val(data, valid ):
    """
    returns one batch of merged data with required IDs from valid
    """
    
    df = pd.DataFrame({'item_id':valid[:,1],'shop_id':valid[:,0]} )
    data = df.merge(data, on=['shop_id','item_id'], how='left').fillna(0)
    return data

def prepare_data_train_LSTM(data, valid, dbn):
    """
    
    """
    train = prepare_train (data, valid)
    lag_cols = []
    for col in data.columns:

        splitted = col.split('$')
        if len(splitted)==1:
            lag_cols.append(col)
            continue
        #if 'shop_item_cnt' not in col:
        #    continue
            
        for db in range(0,dbn-1):
            
            if db == int(splitted[1]):
                lag_cols.append(col)

    X = train[lag_cols]

    
    Y = train[f'shop_item_cnt${dbn-1}']
    
    return X, Y


def prepare_data_validation_LSTM(data, valid, dbn):
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
    Y = test[f'shop_item_cnt${dbn}']
    
    return X, Y

def create_batch_train(merged,shop_item_pairs_WITH_PREV_in_dbn, batch_size, dbn):
    """
    
    """
    #merged = pd.read_csv('data/merged.csv', chunksize=500000)
    #merged = pd.read_csv('data/merged.csv')
    train = np.random.permutation (shop_item_pairs_WITH_PREV_in_dbn[dbn])
    #train = shop_item_pairs_WITH_PREV_in_dbn[dbn]
    chunck_num = (len(train)  // batch_size) + 1
    
    for idx in range(chunck_num):#split shop_item_pairs_WITH_PREV_in_dbn into chuncks
        #for chunck in merged:#split merged into chuncks
        train_ret = prepare_data_train_LSTM(merged,train[idx*batch_size:(idx+1)*batch_size], dbn)
       
        if  train_ret[0].empty:
            yield [None, None]
        
        yield train_ret#, test

def create_batch_val(merged, shop_item_pairs_in_dbn, batch_size, dbn):
    """
    
    """
    #merged = pd.read_csv('data/merged.csv', chunksize=500000) - (DOESNT WORK PROPERLY))))) - use it if merged doesnt fit memory
    #merged = pd.read_csv('data/merged.csv')
    val = shop_item_pairs_in_dbn[dbn]

    shops = np.unique(list(zip(*val))[0])
    items = np.unique(list(zip(*val))[1])

    cartesian_product = np.random.permutation (np.array(np.meshgrid(shops, items)).T.reshape(-1, 2))
    
    chunck_num = (len(cartesian_product)  // batch_size) + 1
    for idx in range(chunck_num):
        #for chunck in merged:
        train_ret = prepare_data_validation_LSTM(merged,cartesian_product[idx*batch_size:(idx+1)*batch_size], dbn)
        #When in batches idx no elements that are in (shop, item) in batch of merged
        if  train_ret[0].empty:
            
            yield [None, None]
        #print(len(train_ret))
        
        yield train_ret#, test

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class CustomLSTM(nn.Module):

    def __init__(self, embedding_dim=1, hidden_dim=64,hidden_linear=64,  target_size=1, N_LEVELS=None, device=None):
        super(CustomLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size = self.hidden_dim,
                            batch_first=True,
                            proj_size=target_size,
                            num_layers=N_LEVELS,
                           device=device)

        self.linear_layers = nn.Sequential(
            nn.Linear(target_size, hidden_linear),
            nn.ReLU(),
            nn.Linear(hidden_linear, 1)
        ).to(device)

        # The linear layer that maps from hidden state space to tag space
        

    def forward(self, data):
        
        lstm_out, _ = self.lstm(data)

        
        linear_out = self.linear_layers(lstm_out)
        
        return linear_out

def preapre_X_LSTM_format(X_train, dbn, device):
    #Batch come here
    #one row - one train example
    cols=[]
    data = defaultdict(list)
    for col in X_train.columns:
        if 'change' in col:
            continue
        #if 'avg_item_price' in col:
        #   cols.append(col)
        #   continue

        if 'shop_item_cnt' in col:
            cols.append(col)
            continue

        
            
        """
        if not col[-1].isdigit():
            continue

        

        if not any(c in col for c in SELECTED_COLUMNS) :
            continue
        """
        
        
    #print('COLUMNS',list(X_train[cols].columns))
    #print(X_train[cols].columns)
    X_train = X_train[cols].values.reshape(len(X_train), dbn-1, -1)
    #print(X_train)
    return torch.tensor(X_train, device=device).to(dtype=torch.float32)


def preapre_Y_LSTM_format(Y_train, device):
    return torch.tensor(Y_train, device=device).to(dtype=torch.float32)

def train_lstm(model=None,shop_item_pairs_WITH_PREV_in_dbn=None,optimizer=None,loss_fn=None,   merged=None,batch_size=None, val_month=None, epochs=None):
    
    first=True
    rmse = 0
    c=0
    
    preds_l=[]
    y_true_l=[]

    grads = []
    for X_train,Y_train  in create_batch_train(merged,shop_item_pairs_WITH_PREV_in_dbn,batch_size, val_month):
        
        if X_train is None:
            print('None')
            continue
        Y_train = np.clip(Y_train,0,20)
        
        if X_train.empty:
            print('None')
            continue
        X_train = make_X_lag_format(X_train, val_month-1)
        X_train=preapre_X_LSTM_format(X_train, val_month-1, device)
        Y_train = preapre_Y_LSTM_format(Y_train, device)
        
        
        optimizer.zero_grad()

        preds = model(X_train)[:,-1,:]
        
        preds_l.append(torch.squeeze(preds))
        y_true_l.append(torch.squeeze(Y_train))
        
        loss_train = loss_fn(torch.squeeze(preds), 
                             torch.squeeze(Y_train))
        
        
        
        loss_train.backward()
        
        total_norm = torch.max( torch.stack([p.grad.detach().abs().max() for p in model.parameters()]) )
        grads.append(total_norm)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        c+=1

    preds_l = torch.concat(preds_l)
    y_true_l = torch.concat(y_true_l)
    #print('mean of max grad,',torch.mean(torch.tensor(grads)))
    with torch.no_grad():
        metric = torch.sqrt(loss_fn(torch.clamp(preds_l,0,20), y_true_l))
        #print(pd.DataFrame(torch.clamp(y_true_l,0,20).numpy(force=True)).describe())
        
    return model, metric

def validate_lstm(model,shop_item_pairs_in_dbn,merged,batch_size, val_month):
    val_error = 0
    c=0
    val_preds=[]
    preds_l=[]
    y_true_l=[]
    #create_batch_train(merged,batch_size, val_month) - return train set, where Y_val
    #is shop_item_cnt_month{val_month}
    loss_fn = nn.MSELoss()

    for X_val, Y_val in create_batch_val(merged,shop_item_pairs_in_dbn,batch_size, val_month):#but then cartesian product used
        
        if X_val is None:
            continue
        if X_val.empty:
            print('None')
            continue
        Y_val = np.clip(Y_val,0,20)        
        X_val = make_X_lag_format(X_val, val_month)
        
        
        X_val=preapre_X_LSTM_format(X_val, val_month, device)
        Y_val = preapre_Y_LSTM_format(Y_val, device)


        with torch.no_grad():
            y_val_pred = model(X_val)[:,-1,:]
            loss_rmse = torch.sqrt(loss_fn(torch.squeeze(y_val_pred), torch.squeeze(Y_val)))
            
            
            preds_l.append(torch.squeeze(y_val_pred))
            y_true_l.append(torch.squeeze(Y_val))
            val_preds.append(y_val_pred)
            c+=1
            
    preds_l = torch.concat(preds_l)
    y_true_l = torch.concat(y_true_l)

    
    with torch.no_grad():
        metric = torch.sqrt(loss_fn(torch.clamp(preds_l,0,20)*20, y_true_l*20))
        #print(pd.DataFrame(torch.clamp(y_true_l,0,20).numpy(force=True)).describe())
        
            
    return preds_l, metric

def validate_LSTM_pipeline(merged=None,
                shop_item_pairs_in_dbn=None, 
                shop_item_pairs_WITH_PREV_in_dbn=None,
                epochs=None,
                start_val_month =None,
                lr=None,
                step_size=None,
                gamma=None,
                EMBEDDING_DIM=None,
                HIDDEN_DIM=None, 
                hidden_linear=None,
                TARGET_SIZE=None,
                N_LEVELS=None,
                device=None,
                batch_size=None):
    """
    Function for validating model
    
    """
    
    val_errors = []
    
    
    for val_month in range(start_val_month, 34):

        
        loss_fn = nn.MSELoss()
        print('date_block_num', val_month)
        print('month', val_month%12)
        
        model = CustomLSTM(embedding_dim=EMBEDDING_DIM,
                   hidden_dim=HIDDEN_DIM,
                   hidden_linear=hidden_linear,
                   target_size=TARGET_SIZE,
                   N_LEVELS=N_LEVELS,
                   device=device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        for epoch in range(epochs):
        
            
            
            model,train_error = train_lstm(model=model,
                                           shop_item_pairs_WITH_PREV_in_dbn=shop_item_pairs_WITH_PREV_in_dbn,
                                           optimizer=optimizer,
                                           loss_fn=loss_fn, 
                                           merged=merged,
                                           batch_size=batch_size, 
                                           val_month=val_month)
            
            scheduler.step()
            if epoch % 3 == 0:
                val_pred, val_error = validate_lstm(model,
                                                    shop_item_pairs_in_dbn,
                                                    merged,
                                                    batch_size,
                                                    val_month)

                print('prediction')
                print('mean', torch.mean(val_pred))
                print('max', torch.max(val_pred))
                print('min', torch.min(val_pred))
                print('std', torch.std(val_pred))
                print('mean train rmse on epoch', epoch,':',train_error )
                print('mean val rmse on epoch', epoch,':',val_error )
                

            if epoch % 13 == 0:
                print('lr:',scheduler.get_last_lr())
                
            
            val_errors.append(val_error)
            val_preds.append(val_pred)
        

    return val_errors, val_preds

def normalize(data):
    data[[col for col in merged.columns if 'cnt' in col ]] = np.clip(data[[col for col in merged.columns if 'cnt' in col ]],0,20)

    logarithmic = np.log(data[[col for col in merged.columns if 'price' in col ]])
    data[[col for col in merged.columns if 'price' in col ]] = np.clip(np.nan_to_num( logarithmic,  posinf=0, neginf=0),0,10000) / 10000
    data[[col for col in merged.columns if 'cnt' in col ]] /= 20

def create_submission(model,shop_item_pairs_in_dbn,merged,batch_size):
    val_month = 34
    test = pd.read_csv('../data_cleaned/test.csv')
    
    data_test = test
    PREDICTION = pd.DataFrame(columns=['shop_id','item_id','item_cnt_month'])
    
    print('date_block_num', val_month)
    print('month', val_month%12)
    for X_val, Y_val in create_batch_val(merged,shop_item_pairs_in_dbn,batch_size, val_month):#but then cartesian product used
        shops= X_val.shop_id
        items = X_val.item_id
        if X_val is None:
            continue
        if X_val.empty:
            print('None')
            continue
        Y_val = np.clip(Y_val,0,20)        
        X_val = make_X_lag_format(X_val, val_month)
        
        
        X_val=preapre_X_LSTM_format(X_val, val_month, device)
        Y_val = preapre_Y_LSTM_format(Y_val, device)

        
        with torch.no_grad():
            y_val_pred = model(X_val)[:,-1,:]
            y_val_pred = y_val_pred.numpy(force=True).flatten()
            y_val_pred = np.clip(y_val_pred*20,0,20)

            app = pd.DataFrame({'item_id':items,
                                'shop_id': shops,
                                'item_cnt_month':y_val_pred})
            PREDICTION = pd.concat([PREDICTION, app],ignore_index=True)


    
    data_test = data_test.merge(PREDICTION,on=['shop_id','item_id'])[['ID','item_cnt_month']]
    return data_test


def create_submission_pipeline (merged=None,
                                shop_item_pairs_in_dbn=None, 
                                shop_item_pairs_WITH_PREV_in_dbn=None,
                                epochs=None,
                                start_val_month =None,
                                lr=None,
                                step_size=None,
                                gamma=None,
                                EMBEDDING_DIM=None,
                                HIDDEN_DIM=None, 
                                hidden_linear=None,
                                TARGET_SIZE=None,
                                N_LEVELS=None,
                                device=None,
                                batch_size=None):
    

    model = CustomLSTM(EMBEDDING_DIM, 
                        HIDDEN_DIM, 
                        hidden_linear,
                        TARGET_SIZE,
                        N_LEVELS,
                        device=device
                        )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    loss_fn = nn.MSELoss()
    
    val_month=34
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    for epoch in range(epochs):
        print('training epoch',epoch)
        model,columns_order = train_lstm(model=model,
                                        shop_item_pairs_WITH_PREV_in_dbn=shop_item_pairs_WITH_PREV_in_dbn,
                                        optimizer=optimizer,
                                        loss_fn=loss_fn, 
                                        merged=merged,
                                        batch_size=batch_size, 
                                        val_month=val_month, 
                                        epochs=epochs
                                        )
        
        scheduler.step()
    
    
    data_test = create_submission(model,shop_item_pairs_in_dbn,merged,batch_size)

    return data_test


if __name__ == '__main__':
    data_train = pd.read_csv('../data_cleaned/data_train.csv')
    test = pd.read_csv('../data_cleaned/test.csv')
    test['date_block_num'] = 34
    data_train = pd.concat([data_train,test ], ignore_index=True).drop('ID', axis=1).fillna(0)

    shop_item_pairs_in_dbn, shop_item_pairs_WITH_PREV_in_dbn = prepare_past_ID_s(data_train)
    
    device='cuda' if torch.cuda.is_available() else 'cpu'
    
    chunksize = 50000
    l=[]
    with pd.read_csv('data/merged.csv', chunksize=chunksize) as reader:
        for chunk in reader:
            l.append(chunk)

    
    merged = pd.concat(l)

    
    EMBEDDING_DIM=1
    HIDDEN_DIM=64#512
    TARGET_SIZE=1
    N_LEVELS=1
    hidden_linear=64
    epochs=30
    start_val_month=22

    batch_size =20000
    val_preds=[]
    lr = 0.003
    step_size=40
    gamma=0.1

    
    normalize(merged)
    """
    val_errors, val_preds=validate_LSTM_pipeline(merged=merged,
                                                shop_item_pairs_in_dbn=shop_item_pairs_in_dbn, 
                                                shop_item_pairs_WITH_PREV_in_dbn=shop_item_pairs_WITH_PREV_in_dbn,
                                                epochs=epochs,
                                                start_val_month =start_val_month,
                                                lr=lr,
                                                step_size=step_size,
                                                gamma=gamma,
                                                EMBEDDING_DIM=EMBEDDING_DIM,
                                                HIDDEN_DIM=HIDDEN_DIM, 
                                                hidden_linear=hidden_linear,
                                                TARGET_SIZE=TARGET_SIZE,
                                                N_LEVELS=N_LEVELS,
                                                device=device,
                                                batch_size=batch_size
                                            )
    """
    submission = create_submission_pipeline(merged=merged,
                                            shop_item_pairs_in_dbn=shop_item_pairs_in_dbn, 
                                            shop_item_pairs_WITH_PREV_in_dbn=shop_item_pairs_WITH_PREV_in_dbn,
                                            epochs=epochs,
                                            start_val_month =start_val_month,
                                            lr=lr,
                                            step_size=step_size,
                                            gamma=gamma,
                                            EMBEDDING_DIM=EMBEDDING_DIM,
                                            HIDDEN_DIM=HIDDEN_DIM, 
                                            hidden_linear=hidden_linear,
                                            TARGET_SIZE=TARGET_SIZE,
                                            N_LEVELS=N_LEVELS,
                                            device=device,
                                            batch_size=batch_size
                                            )
    
    submission.to_csv('submission.csv', index=False)
    