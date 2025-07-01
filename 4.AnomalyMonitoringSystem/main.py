import sys
import os
import argparse
import pandas as pd
import numpy as np 
import importlib  
import requests
import json  
from datetime import date, datetime, timedelta 
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error 

# 使用 Model
level1_model_dir = 'Model'
level2_model_dir = 'Model/NoDecomp'
level2_vmd_model_dir = 'Model/VMD_NoDecomp'


def get_config():
    """取得參數設定
    
    使用範例
        第一階階段當日數據(即時分類) : python main.py --level1 -t --sendNotify
        第一階段指定日期數據 : python main.py --level1 --sdate 2023-04-02 --edate 2024-12-3
        
        第二階段當日數據(預測兩日後) : python main.py --level2 -t
        第二階段論文資料集一次性預測 : python main.py --level2 --paper
        第二階段指定日期數據 : python main.py --level2 --sdate 2024-04-28 --edate 2024-12-3  
    """
    
    parser = argparse.ArgumentParser(description="取得參數設定")  
    parser.add_argument('--level1', action='store_true', help="是否為第一階段")
    parser.add_argument('--level2', action='store_true', help="是否為第二階段")
    parser.add_argument('-t', '--today', action='store_true', help="是否為今日數據")
    parser.add_argument('--sdate', type=str, required=False, help="若不是今日數據，請輸入開始日期 ex. 2023-04-02")
    parser.add_argument('--edate', type=str, required=False, help="若不是今日數據，請輸入結束日期 ex. 2024-10-31") 
    parser.add_argument('--paper', action='store_true', help="(僅第一次執行) 是否為論文資料集一次性預測，2024-04-30 之前")
    parser.add_argument('--show', action='store_true', help="是否顯示圖表")
    parser.add_argument('--sendNotify', action='store_true', help="是否發送通知")
    
    args = parser.parse_args()
    
    return args


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
label_name = 'GroupCount'

rootPath = os.getcwd()
lib_path = f'{rootPath}\\Lib\\'  
print(f'lib_path : {lib_path}') 

if lib_path not in sys.path: sys.path.append(lib_path)
     
     
def load_lib(lib_name):
    """動態載入指定的 lib"""
    
    if lib_name not in sys.modules:
        importlib.import_module(lib_name)
    else:
        importlib.reload(sys.modules[lib_name])


def get_db_object():
    """取得資料庫物件"""
    
    load_lib('Database')
    db = getattr(sys.modules['Database'], 'Database', None) 
     
    with open('appsettings.json') as json_file:
        json_result = json.load(json_file)
     
    database = db(
        server = json_result['ProdDatabaseConfig']['ServerName'], 
        database = json_result['ProdDatabaseConfig']['DatabaseName'], 
        username = json_result['ProdDatabaseConfig']['Account'], 
        password = json_result['ProdDatabaseConfig']['Password']
    )
    
    return database


def get_date_range(sdate_str, edate_str):
    sdate = datetime.strptime(sdate_str, "%Y-%m-%d").date()
    edate = datetime.strptime(edate_str, "%Y-%m-%d").date()
    
    # Generate list of dates between sdate and edate
    date_list = [sdate + timedelta(days=x) for x in range((edate - sdate).days + 1)]
    date_list = [current_date.strftime("%Y-%m-%d") for current_date in date_list]

    return date_list


def run_level1(database, classifier, yyyymmdd):
    """第一階段分類"""
    iislog_csv = f'Data/iis_{yyyymmdd.replace('-', '')}.csv'
    print(f'\033[93m# 第一階段分類 {yyyymmdd}\x1b[0m')
    
    # 檢查檔案室否存在，紅色字體警告
    if not os.path.isfile(iislog_csv): 
        print(f'\033[91m{iislog_csv} 檔案不存在\x1b[0m') 
        
        print('Step1. 建立空的 DataFrame')
        df = pd.DataFrame(columns=['RequestTime', 'Type', 'SendBytes', 'ReceiveBytes', 'TimeTaken', 'GroupCount']) 
        
        print('Step2. 產生空白的資料')
        types = ['1', '2', '3', '4', '5', '6', '246'] 
        for t in types:
            df.loc[len(df)] = [f'{yyyymmdd}', t, 0, 0, 0, 0]
        
    else:   
        print('Step1. 讀取 IIS Log 的 csv 檔案到 pandas DataFrame')
        df = classifier.load_datasource(iislog_csv, yyyymmdd = yyyymmdd)
        # print(df.head()) 
        
        print('Step2. 載入模型')
        model = classifier.load_model()
        
        print('Step3. 轉換成預測用的特徵')  
        features = classifier.convert_features(df) 
    
        print('Step4. 預測')
        y_pred = classifier.predict(model, features) 
        
        print('Step5. 將預測結果加入原始資料集')
        df = classifier.append_predicted(df, y_pred) 
        
        print('Step6. 加總成每天的數據')    
        df = classifier.group_by_day(df) 
    
    print('Step7. 將預測結果寫入資料庫')
    database.insert_to_predictions(df)


def config_leve2(type, label_name):  
    """第二階段設定""" 
    scaler_filename = f'{label_name}_mt1_data{type}_batch14'
    model_file_name = f'{label_name}_mt1_data{type}_batch14.model'
    
    load_lib('IISLogTrafficPredictor')
    lib = getattr(sys.modules['IISLogTrafficPredictor'], 'IISLogTrafficPredictor', None)
    predictor = lib(model_dir = level2_model_dir, is_check_version = True)
         
    return scaler_filename, model_file_name, predictor

     
# 預測第二階段(整批預測並整批寫回，訓練模式，已經知道答案的情況)
def run_level2_history_predict(database, current_date, type, label_name):
    print('\033[93m# 第二階段預測(整批預測)\x1b[0m')
    scaler_filename, model_file_name, predictor = config_leve2(type, label_name) 
    
    print('Step1. 讀取資料庫資料') 
    # data_source = predictor.load_data(label_name, types, show_predict_chart=False) 
    # data_source = data_source[-394:]  
    df = database.query_top_n_data(date = current_date, n = 394, type = f'{type}') 
   
    print('Step2. 填充缺失的數據')
    df = database.fill_groupcount_with_median(df, label_name = label_name, days_range = 30)  
    data_source = df[label_name].values 
      
    print('Step3. 正規化資料')
    x_norm, _, _, y_scaler = \
        predictor.normalize(data_source, scaler_filename = scaler_filename)
 
    print('Step4. 轉換資料為 LSTM 格式')
    x_lstm, y_lstm =  predictor.reshape_data_for_history_lstm(data = x_norm) 
         
    print('Step5. 預測資料')
    predict_result = predictor.predict(model_file_name, x_lstm)
     
    print('Step6. 還原預測資料')
    prediction_data, actual_data  = predictor.inverse_transform_and_reshape_helper(
        y_predict = predict_result, 
        y_scaler = y_scaler,
        y_lstm_actual = y_lstm
    )   
    
    print('\033[93m# 第二階段數據驗證(整批預測)\x1b[0m')
    print('Step1. 轉換資料為 LSTM 格式，目的是為了取得日期資料 & y_data')
    x_data, y_data, date_data = predictor.reshape_data_for_history_lstm2(data_frame = df, label_name = label_name) 
     
    print('Step2. 計算 MAE & MAPE')
    mae, mape = predictor.calculate_mae_mape(y_data, prediction_data) 
    print(f'\tMAE: {mae}, MAPE: {mape}')
     
    print('Step3. 將預測結果寫回資料庫')
    database.insert_to_traffic_predictions(
        type = type, 
        label_name = label_name,
        predicted_label_name = f'{label_name}Predicted',
        date_data = date_data,
        prediction_data = prediction_data,
    ) 
    
    # print('Step4. 顯示預測圖表')
    # threshold_up, threshold_down  = database.get_threadhold(type)
    # database.show_predict_chart(actual_data, prediction_data, threshold_up, threshold_down, label_name)
    # print(date_data.reshape(-1))
    


# 預測第二階段(整批預測但寫回最後一筆，正式模式，未知答案的情況)
def run_level2_prod_predict(database, current_date, type, label_name):
    print('\033[93m# 第二階段預測(正式預測)\x1b[0m')
    scaler_filename, model_file_name, predictor = config_leve2(type, label_name) 
    
    # 預測後天的資料 
    feature_date = datetime.strptime(current_date, "%Y-%m-%d").date() + timedelta(days=2)
    print(f'輸入日期: {current_date}, 預測日期: {feature_date}') 
    
    print('Step1. 讀取資料庫資料')  
    df = database.query_top_n_data(date = current_date, n = 394, type = type)   
    
    print('Step2. 填充缺失的數據')
    df = database.fill_groupcount_with_median(df, label_name = label_name, days_range = 30)  
    data_source = df[label_name].values 
    
    print('Step3. 正規化資料')
    x_norm, _, _, y_scaler = \
        predictor.normalize(data_source, scaler_filename = scaler_filename) 
    
    print('Step4. 轉換資料為 LSTM 格式')
    x_lstm =  predictor.reshape_data_for_production_lstm(data = x_norm) 
    
    print('Step5. 預測資料')
    predict_result = predictor.predict(model_file_name, x_lstm) 
     
    print('Step6. 還原預測資料')
    prediction_data, _ = predictor.inverse_transform_and_reshape_helper(
        y_predict = predict_result, 
        y_scaler = y_scaler
    )  
    
    print('Step7. 將預測結果寫回資料庫(僅寫回最後一筆)') 
    print(f'寫回日期: {feature_date}, 預測數據: {prediction_data[-1]}')  
    database.insert_to_traffic_predictions(
        type = type, 
        label_name = label_name,
        predicted_label_name = f'{label_name}Predicted',
        date_data = np.array(feature_date),
        prediction_data = np.array(prediction_data[-1]),
    )


def config_leve2_vmd(type, label_name):   
    scaler_filename = f'{label_name}_mt1_data{type}_batch14'
    model_file_name = f'{label_name}_mt1_data{type}_batch14.model'
    
    load_lib('IISLogTrafficPredictor')
    lib = getattr(sys.modules['IISLogTrafficPredictor'], 'IISLogTrafficPredictor', None)
    predictor = lib(model_dir = level2_vmd_model_dir, is_check_version = True)
    
    return scaler_filename, model_file_name, predictor

     
# 預測第二階段(整批預測並整批寫回，訓練模式，已經知道答案的情況)
def run_level2_history_vmd_predict(database, current_date, type, label_name):
    print('\033[93m# 第二階段 VMD 預測(整批預測)\x1b[0m')
    scaler_filename, model_file_name, predictor = config_leve2_vmd(type, label_name) 
    
    print('Step1. 讀取資料庫資料')  
    df = database.query_top_n_data(date = current_date, n = 394, type = f'{type}') 
   
    print('Step2. 填充缺失的數據')
    df = database.fill_groupcount_with_median(df, label_name = label_name, days_range = 30)  
    data_source = df[label_name].values 
    
    print('Step3. VMD 分解(前處理)') 
    u = predictor.vmd_decomposition(data_source, K = 8) 
    
    print('Step4. 異常資料前處理')
    processed_signal = predictor.preprocess_smooth_signal_imfs(u, sigma=1, std_threshold=1) 
    original_signal = np.sum(u, axis=0)
    
    mae = mean_absolute_error(data_source, processed_signal)
    mape = mean_absolute_percentage_error(data_source, processed_signal)
    print(f'\t 原始資料 MAE: {mae}, MAPE: {mape}') 
    mae = mean_absolute_error(original_signal, processed_signal)
    mape = mean_absolute_percentage_error(original_signal, processed_signal) 
    print(f'\t VMD 資料 MAE: {mae}, MAPE: {mape}')
 
    print('Step5. 正規化資料')
    x_norm, _, _, y_scaler = \
        predictor.normalize(processed_signal, scaler_filename = scaler_filename)
 
    print('Step6. 轉換資料為 LSTM 格式')
    x_lstm, y_lstm =  predictor.reshape_data_for_history_lstm(data = x_norm) 
         
    print('Step7. 預測資料')
    predict_result = predictor.predict(model_file_name, x_lstm)
     
    print('Step8. 還原預測資料')
    prediction_data, actual_data  = predictor.inverse_transform_and_reshape_helper(
        y_predict = predict_result, 
        y_scaler = y_scaler,
        y_lstm_actual = y_lstm
    )   
    
    print('\033[93m# 第二階段數據驗證(整批預測)\x1b[0m')
    print('Step1. 轉換資料為 LSTM 格式，目的是為了取得日期資料 & y_data')
    x_data, y_data, date_data = predictor.reshape_data_for_history_lstm2(data_frame = df, label_name = label_name) 
     
    print('Step2. 計算 MAE & MAPE')
    mae, mape = predictor.calculate_mae_mape(y_data, prediction_data) 
    print(f'\tMAE: {mae}, MAPE: {mape}')
     
    print('Step3. 將預測結果寫回資料庫')
    database.insert_to_traffic_predictions(
        type = type, 
        label_name = label_name,
        predicted_label_name = f'{label_name}Predicted2',
        date_data = date_data,
        prediction_data = prediction_data,
    ) 
     


# 預測第二階段(整批預測但寫回最後一筆，正式模式，未知答案的情況)
def run_level2_prod_vmd_predict(database, current_date, type, label_name):
    print('\033[93m# 第二階段 VMD 預測(正式預測)\x1b[0m')
    scaler_filename, model_file_name, predictor = config_leve2(type, label_name) 
    
    # 預測後天的資料 
    feature_date = datetime.strptime(current_date, "%Y-%m-%d").date() + timedelta(days=2)
    print(f'輸入日期: {current_date}, 預測日期: {feature_date}') 
    
    print('Step1. 讀取資料庫資料')  
    df = database.query_top_n_data(date = current_date, n = 394, type = type)   
    
    print('Step2. 填充缺失的數據')
    df = database.fill_groupcount_with_median(df, label_name = label_name, days_range = 30)  
    data_source = df[label_name].values 
    
    print('Step3. VMD 分解(前處理)') 
    u = predictor.vmd_decomposition(data_source, K = 8) 
    
    print('Step4. 異常資料前處理')
    processed_signal = predictor.preprocess_smooth_signal_imfs(u, sigma=1, std_threshold=1) 

    print('Step5. 正規化資料')
    x_norm, _, _, y_scaler = \
        predictor.normalize(processed_signal, scaler_filename = scaler_filename) 
    
    print('Step6. 轉換資料為 LSTM 格式')
    x_lstm =  predictor.reshape_data_for_production_lstm(data = x_norm) 
    
    print('Step7. 預測資料')
    predict_result = predictor.predict(model_file_name, x_lstm) 
     
    print('Step8. 還原預測資料')
    prediction_data, _ = predictor.inverse_transform_and_reshape_helper(
        y_predict = predict_result, 
        y_scaler = y_scaler
    )  
    
    print('Step9. 將預測結果寫回資料庫(僅寫回最後一筆)') 
    print(f'寫回日期: {feature_date}, 預測數據: {prediction_data[-1]}')  
    database.insert_to_traffic_predictions(
        type = type, 
        label_name = label_name,
        predicted_label_name = f'{label_name}Predicted2',
        date_data = np.array(feature_date),
        prediction_data = np.array(prediction_data[-1]),
    )

    
def show_chart(database, type_str): 
    df = database.query_top_n_data(date = '2024-10-30', n = 600, type = type_str)  
    # df = database.query_top_n_data(date = '2024-04-30', n = 379, type = type)  
    actual_data = df[label_name].values 
    prediction_data = df[f'{label_name}Predicted'].values
    prediction_data2 = df[f'{label_name}Predicted2'].values
    date_data = df['RequestTime'].values
    
    
    threshold_up, threshold_down  = database.get_threadhold(type_str, is_vmd = False)
    threshold_up2, threshold_down2  = database.get_threadhold(type_str, is_vmd = True)
    
    # 單一圖表顯示
    # database.show_predict_chart(actual_data, prediction_data, threshold_up, threshold_down, label_name, date_data) 
    # database.show_predict_chart(actual_data, prediction_data2, threshold_up2, threshold_down2, label_name, date_data)
    
    database.show_predict_chart2(actual_data, type_str,
        prediction_data, threshold_up, threshold_down, 
        prediction_data2, threshold_up2, threshold_down2, 
        label_name, date_data
    )
    
    
def send_notification(): 
    with open('appsettings.json') as json_file:
        json_result = json.load(json_file)
     
    api_url = json_result['MockRequest']['ApiUrl']
    api_token = json_result['MockRequest']['ApiToken']
     
    headers = {
        'Authorization': f'Bearer {api_token}'
    }
    
    response = requests.get(api_url, headers=headers)

    if response.status_code != 200:
        print("通知發送失敗({response.status_code}) :: {response.text}")
    else:
        print("通知發送成功")
        


if __name__ == '__main__':  
    config = get_config()  
    database = get_db_object()
    
    
    if config.level1: 
        load_lib('IISLogTypeClassification')
        lib = getattr(sys.modules['IISLogTypeClassification'], 'IISLogTypeClassification', None)
        classifier = lib(model_dir = level1_model_dir) 
        
        if config.today:
            # 當日數據
            yesterday = datetime.now() - timedelta(days=1)
            print(f'第一階段分類: 處理昨日 {yesterday.strftime("%Y-%m-%d")} 數據')
            run_level1(database, classifier, yyyymmdd=yesterday.strftime('%Y-%m-%d'))
            
            current_date = datetime.now().strftime('%Y-%m-%d')
            print(f'第一階段分類: 處理今日 {current_date} 數據')
            run_level1(database, classifier, yyyymmdd=f'{current_date}')
        else:
            # 歷史數據
            date_list = get_date_range(config.sdate, config.edate)
            print(f'第一階段分類: 處理日期範圍 {config.sdate} ~ {config.edate} 數據')
            for current_date in date_list: 
                run_level1(database, classifier, yyyymmdd=f'{current_date}')
        
        
    if config.level2:
        types = ['1', '246', '3', '5']
        
        if config.today:
            # 當日數據
            yesterday = datetime.now() - timedelta(days=1)
            print(f'第二階段預測: 處理昨日 {yesterday.strftime("%Y-%m-%d")} 數據')
            for type in types:
                # 非 VMD 數據
                run_level2_prod_predict(
                    database = database,
                    current_date = yesterday.strftime('%Y-%m-%d'), 
                    type = type, 
                    label_name = label_name
                )
                # VMD 數據
                run_level2_prod_vmd_predict(
                    database = database,
                    current_date = yesterday.strftime('%Y-%m-%d'), 
                    type = type, 
                    label_name = label_name
                )
            
            current_date = datetime.now().strftime('%Y-%m-%d')
            print(f'第二階段預測: 處理今日 {current_date} 數據')
            for type in types:
                # 非 VMD 數據
                run_level2_prod_predict(
                    database = database,
                    current_date = current_date, 
                    type = type, 
                    label_name = label_name
                )
                # VMD 數據
                run_level2_prod_vmd_predict(
                    database = database,
                    current_date = current_date, 
                    type = type, 
                    label_name = label_name
                ) 
        else:
            if config.paper:
                print('第二階段預測: 論文資料集一次性預測，2024-04-30 之前的數據')
                for type in types:
                    # 論文資料集預測 (非 VMD 數據)
                    run_level2_history_predict(
                        database = database,
                        current_date = '2024-04-30', 
                        type = type, 
                        label_name = label_name
                    ) 
                    # 論文資料集預測 (VMD 數據)
                    run_level2_history_vmd_predict(
                        database = database,
                        current_date = '2024-04-30', 
                        type = type, 
                        label_name = label_name
                    )
            else: 
                # 回測數據
                date_list = get_date_range(config.sdate, config.edate)
                print(f'第二階段預測: 處理日期範圍 {config.sdate} ~ {config.edate} 數據')
                for type in types:
                    for current_date in date_list: 
                        # 非 VMD 數據
                        run_level2_prod_predict(
                            database = database,
                            current_date = current_date, 
                            type = type, 
                            label_name = label_name
                        )
                        # VMD 數據
                        run_level2_prod_vmd_predict(
                            database = database,
                            current_date = current_date, 
                            type = type, 
                            label_name = label_name
                        )
         
         
    if config.show:
        show_chart(database, type_str = '1') 


    if config.sendNotify: 
        send_notification()