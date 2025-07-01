import joblib 
import importlib
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import keras
import os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error 
from vmdpy import VMD
from scipy.ndimage import gaussian_filter1d 

class IISLogTrafficPredictor:
    def __init__(self, model_dir, is_check_version = True): 
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  
        
        self.model_dir = model_dir
        
        if is_check_version: 
            print('檢查套件版本')
            self.__check_version('sklearn', '1.5.2')
            self.__check_version('tensorflow', '2.17.0')
            self.__check_version('keras', '3.6.0')
        
         
    def __check_version(self, package, version):
        pkg = importlib.import_module(package)
        print(f'\t{package} version: {pkg.__version__}')
        if pkg.__version__ != version:
            raise Exception(f'{package} version is not {version}')
     

    def load_data(self, label_name, predicteds, show_predict_chart = False):
        """
        (測試用)載入資料
        
        Args:
            label_name (str): 標籤名稱
            predicteds (list): 預測值
            show_predict_chart (bool): 是否顯示預測圖表
            
        Returns:
            numpy.ndarray: 資料
        """
        df_source = pd.read_csv('iislog_level2_datasource.csv')
        df_source = df_source[df_source['Predicted'].isin(predicteds)]
        data = df_source[label_name].values
        
        if show_predict_chart:
            plt.figure(figsize=(10, 2))
            plt.plot(data, label='實際')
            plt.show() 
            
        return data
    
    def vmd_decomposition(self, data, K = 8):
        """
        對資料進行 VMD 分解
        
        Args:
            data (numpy.ndarray): 資料
            K (int): 要分解的模態數量
            
        Returns:
            numpy.ndarray: 分解後的 IMFs
        """
        alpha = 1          # 平滑性控制參數
        tau = 0.               # 二次懲罰因子
        # K = 7                 # 要分解的模態數量
        DC = 0                # 是否保持第一個模態為DC(直流分量)
        init = 1              # 初始化模式
        tol = 1e-6            # 收斂容忍度

        
        # 進行 VMD 分解
        u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)

        # u 是分解後的 IMFs (形狀為 K x len(data))
        # K, N = u.shape  # K 是 IMFs 的數量, N 是每個 IMF 的長度
        
        return u


    def preprocess_smooth_signal_imfs(self, u, sigma=1, std_threshold=1):
        """
        對 IMFs 進行異常檢測和平滑處理
        
        Args:
            u (numpy.ndarray): IMFs
            sigma (float): 高斯平滑的標準差
            std_threshold (float): 標準差倍數
            
        Returns:
            numpy.ndarray: 處理過的信號
        """
        
        K, N = u.shape
        print(f'IMFs 的數量: {K}, 每個 IMF 的長度: {N}')
        
        # 創建一個新的矩陣來存儲處理後的 IMFs
        processed_u = u.copy()

        # 逐一對每個 IMF 進行異常檢測和平滑處理
        for i in range(K):
            mean_val = np.mean(u[i])
            std_val = np.std(u[i])
            
            # 找出異常點（超過標準差倍數的部分）
            anomaly_mask = np.abs(u[i] - mean_val) > std_threshold * std_val
            # print(f'將異常部分的值進行平滑處理，imf {i + 1} 有 {np.sum(anomaly_mask)} 個異常點')
            
            if np.any(anomaly_mask):  
                # 使用高斯平滑
                smoothed_anomalies = gaussian_filter1d(u[i][anomaly_mask], sigma=sigma)  
                
                # 更新處理後的 IMF 中的異常點
                processed_u[i][anomaly_mask] = smoothed_anomalies

        # 將處理過的 IMFs 重組以重建信號
        processed_signal = np.sum(processed_u, axis=0)

        return processed_signal
    
    
    def split_train_test_data(self, x, y, split_rate = 0.9):   
        """
        (測試用)分割訓練集和測試集
        
        Args:
            x (numpy.ndarray): x 特徵資料
            y (numpy.ndarray): y 目標資料
            split_rate (float): 分割比例
            
        Returns:
            numpy.ndarray: x 訓練集
            numpy.ndarray: y 訓練集
            numpy.ndarray: x 測試集
            numpy.ndarray: y 測試集
        """
        # 計算要分割的資料點的索引
        split_index = int(x.shape[0] * split_rate)
        
        # 將資料分割為訓練集和測試集
        x_train, x_test = x[:split_index], x[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
    
        return x_train, y_train, x_test, y_test

 
    def normalize(self, data, scaler_filename, no = ''):  
        """
        正規化資料
        
        Args:
            data (numpy.ndarray): 資料
            scaler_filename (str): 正規化器檔名
            no (str): 模型編號(訊號分解才需使用此參數)
        
        Returns:
            numpy.ndarray: 正規化後的 x 特徵資料
            numpy.ndarray: 正規化後的 y 目標資料
            sklearn.preprocessing.MinMaxScaler: x 特徵的正規化器
            sklearn.preprocessing.MinMaxScaler: y 目標的正規化器
        """
        x_scaler = joblib.load(f'{self.model_dir}/{scaler_filename}.x_scaler{no}.pkl') 
        y_scaler = joblib.load(f'{self.model_dir}/{scaler_filename}.y_scaler{no}.pkl')

        # print(f'x_scaler.min: {x_scaler.data_min_}, x_scaler.max: {x_scaler.data_max_}') 
        # print(f'y_scaler.min: {y_scaler.data_min_}, y_scaler.max: {y_scaler.data_max_}')
        
        x_norm = x_scaler.transform(data.reshape(-1, 1))
        y_norm = y_scaler.transform(data.reshape(-1, 1))

        return x_norm, y_norm, x_scaler, y_scaler
     
     
    def reshape_data_for_history_lstm(self, data, past_day = 14, future_day = 1, is_history_mode = True):
        """
        將資料轉換為 LSTM 的格式
        
        Args:
            data (numpy.ndarray): 資料
            past_day (int): 過去幾天
            future_day (int): 未來幾天
            is_history_mode (bool): 是否為歷史模式(正式模式因為未知答案，所以整理方式不一樣)
            
        Returns:
            numpy.ndarray: LSTM 格式的 x 特徵資料
            numpy.ndarray: LSTM 格式的 y 目標資料(歷史模式才有)
        """
        x_lstm, y_lstm = [], []
        
        # index_day = 0 # 前幾天要包含當日 1，若不包含當日，則輸入 0
        # 4/4 日為例
        # ex. 含當日，4/2、4/3、4/4，預測 4/5
        # ex. 不含當日，4/2、4/3，預測 4/5
        target_day = past_day + future_day # 預測第 16 天（不包含當天） 
        if is_history_mode: target_day = target_day + 1
        
        # 迴圈遍歷資料，確保不包含當日，且正確預測未來的第 16 天
        for i in range(len(data) - past_day - (target_day - past_day) + 1):
            # 取過去 1~14 天作為輸入
            input_sequence = data[i:i + past_day]
            x_lstm.append(input_sequence)
            
            if is_history_mode:
                target = np.array(data[i + target_day - 1])
                y_lstm.append(target.reshape(1))
        
        if is_history_mode:
            return np.array(x_lstm), np.array(y_lstm)
        else:
            return np.array(x_lstm)


    def reshape_data_for_history_lstm2(self, data_frame, label_name, past_day = 14, future_day = 1, is_history_mode = True):
        """
        將資料轉換為 LSTM 的格式
        
        Args:
            data_frame (pandas.DataFrame): 資料集
            label_name (str): 標籤名稱
            past_day (int): 過去幾天
            future_day (int): 未來幾天
            is_history_mode (bool): 是否為歷史模式(正式模式因為未知答案，所以整理方式不一樣)
            
        Returns:
            numpy.ndarray: LSTM 格式的 x 特徵資料
            numpy.ndarray: LSTM 格式的 y 目標資料 (歷史模式才有)
            numpy.ndarray: LSTM 格式的 y 日期資料
        """
        x_lstm, y_lstm, date_lstm = [], [], []
        
        target_day = past_day + future_day 
        if is_history_mode: target_day = target_day + 1
        
        for i in range(len(data_frame) - past_day - (target_day - past_day) + 1):
            # 取過去 1~14 天作為輸入
            input_sequence = data_frame[label_name][i:i + past_day]
            x_lstm.append(input_sequence)
            
            if is_history_mode:
                target = np.array(data_frame[label_name][i + target_day - 1])
                y_lstm.append(target.reshape(1))
            
            date = np.array(data_frame['RequestTime'][i + target_day - 1])
            date_lstm.append(date.reshape(1))
            
        if is_history_mode:
            return np.array(x_lstm), np.array(y_lstm), np.array(date_lstm)
        else:
            return np.array(x_lstm), np.array(date_lstm)
    
    def reshape_data_for_production_lstm(self, data, past_day = 14):
        return self.reshape_data_for_history_lstm(data, past_day, future_day = 0, is_history_mode = False)
        
    def reshape_data_for_production_lstm2(self, data_frame, label_name, past_day = 14):
        return self.reshape_data_for_history_lstm2(data_frame, label_name, past_day, future_day = 0, is_history_mode = False)
    
    
    def predict(self, model_filename, x_lstm_formate_data, no = ''):
        """
        預測資料
        
        Args:
            model_filename (str): 模型檔名
            x_lstm_formate_data (numpy.ndarray): LSTM 格式的資料
            no (str): 模型編號(訊號分解才需使用此參數)
            
        Returns:
            numpy.ndarray: 預測資料
        """
        model = keras.models.load_model(f'{self.model_dir}/{model_filename}{no}.keras') 
         
        # return model.predict(x_lstm_formate_data, verbose=0)
        return model(x_lstm_formate_data)
 
   
    def calculate_mae_mape(self, y_actual, y_predict):
        """
        計算 MAE 和 MAPE
        
        Args:
            y_actual (numpy.ndarray): 實際資料
            y_predict (numpy.ndarray): 預測資料
            
        Returns:
            float: MAE
            float: MAPE
        """
        mae = mean_absolute_error(y_actual, y_predict)
        mape = mean_absolute_percentage_error(y_actual, y_predict)
        
        return mae, mape
    
    
    def inverse_transform_and_reshape_helper(self, y_predict, y_scaler, y_lstm_actual = None, future_day = 1): 
        """
        將預測資料還原為原始格式
        
        Args:
            y_predict (numpy.ndarray): 預測資料
            y_scaler (sklearn.preprocessing.MinMaxScaler): y 軸的正規化器
            y_lstm_actual (numpy.ndarray): LSTM 實際資料
            future_day (int): 未來幾天的資料
            
        Returns:
            numpy.ndarray: 預測資料
            numpy.ndarray: 實際資料
        """
        prediction_data = y_scaler.inverse_transform(y_predict)
        
        if y_lstm_actual is None:
            return prediction_data.reshape(-1), None
        
        # 還原實際資料還原回原本的結構
        actual = y_lstm_actual.reshape(-1, future_day)  
        actual_data = y_scaler.inverse_transform(actual) 
        
        return prediction_data.reshape(-1), actual_data.reshape(-1) 
 