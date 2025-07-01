import os
import pandas as pd
import numpy as np 
import json

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, Flatten, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from PyEMD import EMD, EEMD , Visualisation

from vmdpy import VMD
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class LstmLib:
    def __init__(self):
        self.model_loss = 'mse' # 模型損失函數
        self.model_metrics = ['mean_absolute_percentage_error'] # 模型評估指標
 
        self.epochs = 100  # 訓練迭代次數


    def reset_data(self, model_dir, labels, model_type, predicteds, batch_size):
        """重設資料
        
        Args:
            model_dir (str): 模型目錄
            labels (str): 標籤
            model_type (int): 模型類型
            predicteds (str): 預設類別
            batch_size (int): 批次大小

        Returns:
            str: 特徵縮放器檔案名稱
            str: 標籤縮放器檔案名稱
            str: 模型檔案名稱
            str: 結果檔案名稱
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        x_scaler_file_name = self.__file_name_replace(f'{model_dir}/{labels}_mt{model_type}_data{predicteds}_batch{batch_size}.x_scaler.pkl')
        y_scaler_file_name = self.__file_name_replace(f'{model_dir}/{labels}_mt{model_type}_data{predicteds}_batch{batch_size}.y_scaler.pkl')
        model_file_name = self.__file_name_replace(f'{model_dir}/{labels}_mt{model_type}_data{predicteds}_batch{batch_size}.model.keras')
        result_file_name = self.__file_name_replace(f'{model_dir}/{labels}_mt{model_type}_data{predicteds}_batch{batch_size}.result.json')

        print(f'x_scaler_file_name: {x_scaler_file_name}')
        print(f'y_scaler_file_name: {y_scaler_file_name}')
        print(f'model_file_name: {model_file_name}')
        print(f'result_file_name: {result_file_name}')

        # 如果檔案存在則刪除 
        self.del_file(x_scaler_file_name)
        self.del_file(y_scaler_file_name)
        self.del_file(model_file_name)
        self.del_file(result_file_name)

        return x_scaler_file_name, y_scaler_file_name, model_file_name, result_file_name


    def __file_name_replace(self, file_name):
        file_name = file_name.replace('[', '')
        file_name = file_name.replace(']', '')
        file_name = file_name.replace('\'', '')
        file_name = file_name.replace(',', '')
        file_name = file_name.replace(' ', '')

        return file_name
    
    def del_file(self, file_name):
        if os.path.exists(file_name):
            os.remove(file_name)
            print(f'檔案 {file_name} 存在，已刪除')


    def load_data(self, predicteds):
        """載入數據

        Args:
            predicteds (array integer): 預設類別

        Returns:
            DataFrame: 數據集
        """
        print('\033[93m# 載入數據\x1b[0m')

        # 如果檔案 iislog_level2_datasource.csv 不存在，則下載
        file_name = 'iislog_level2_datasource.csv'
        if not os.path.exists(file_name):
            import requests
            resp = requests.get('https://raw.githubusercontent.com/lawrence8358/Test/refs/heads/main/iislog_level2_datasource.csv')
            with open(file_name, 'w') as f:
                f.write(resp.content.decode('utf-8'))
    
            # !wget https://raw.githubusercontent.com/lawrence8358/Test/refs/heads/main/iislog_level2_datasource.csv

        df_source = pd.read_csv(file_name)
        df_source = df_source[df_source['Predicted'].isin(predicteds)]

        # 根據 RequestTime 分組，並將其他欄位加總
        df_source = df_source.groupby('RequestTime').sum().reset_index()
        df_source['Predicted'] = f'{predicteds}'

        return df_source


    def vmd_decomposition(self, data, K = 7):
        """
        對信號進行 VMD 分解。

        Args: 
            data (numpy.ndarray): 要分解的信號。
            K (int): 要分解的模態數量。
            
        Returns:
            numpy.ndarray: 分解後的 IMFs。
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


    def preprocess_smooth_signal_imfs(self, u, sigma=2, std_threshold=1):
        """ 將 VMD IMFs，進行異常檢測並平滑處理

        Args:
            u (numpy.ndarray): 2D 的 IMFs 陣列，每一列代表一個 IMF。
            sigma (int): 高斯平滑的標準差。
            std_threshold (float): 異常檢測的標準差倍數，超過這個倍數的值會被認為是異常。
            
        Returns:
            numpy.ndarray: 經過異常平滑處理後的重建信號。
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
            print(f'將異常部分的值進行平滑處理，imf {i + 1} 有 {np.sum(anomaly_mask)} 個異常點')
            
            if np.any(anomaly_mask):  
                # 使用高斯平滑
                smoothed_anomalies = gaussian_filter1d(u[i][anomaly_mask], sigma=sigma)  
                
                # 使用 Savitzky-Golay 濾波器:
                # smoothed_anomalies = savgol_filter(u[i][anomaly_mask], window_length=5, polyorder=2)

                # 更新處理後的 IMF 中的異常點
                processed_u[i][anomaly_mask] = smoothed_anomalies

        # 將處理過的 IMFs 重組以重建信號
        processed_signal = np.sum(processed_u, axis=0)

        return processed_signal


    def plot_imfs_with_original(self, data, imfs):
        """
        繪製原始訊號和每個 IMF 的圖表。

        Args:
            data_source (numpy.ndarray): 原始訊號。
            imfs (numpy.ndarray): 2D 陣列，每一列代表一個 IMF。
        """
        # 創建圖表，IMFs 的數量加上原始訊號
        fig, axs = plt.subplots(imfs.shape[0] + 1, 1, figsize=(10, 2 * (imfs.shape[0] + 1)))

        # 繪製原始訊號
        axs[0].plot(data, 'r')
        axs[0].set_title("Original Signal")
        axs[0].grid(True)

        # 繪製每個 IMF
        for i, imf in enumerate(imfs):
            axs[i + 1].plot(imf)
            axs[i + 1].set_title(f"IMF {i + 1}")
            axs[i + 1].grid(True)
    
        # 顯示圖表
        plt.tight_layout()
        plt.show()


    # 訊號分解 function
    def decomposition(self, data, is_emd = False): 
        """分解時序數據

        Args:
            data (array): 時序數據
            isEMD (bool, optional): 是否使用 EMD. Defaults to False.
        
        Returns:
            eIMFs (array):分解後的 IMFs
            residue (array):分解後的殘差
            imfs (array):包含殘差的 IMFs
        """ 
        print('\033[93m# 訊號分解\x1b[0m')

        if is_emd:
            print("EMD")
            emd = EMD()
            imfs = emd(data.reshape(-1))
        else:
            print("EEMD")
            eemd = EEMD()
            imfs = eemd.eemd(data.reshape(-1))

        # 獲取 IMFs 和殘差
        if is_emd:
            eIMFs, residue = emd.get_imfs_and_residue()
        else:
            eIMFs, residue = eemd.get_imfs_and_residue()

        print(f'IMFs shape : {imfs.shape}')
        print(f'eIMFs shape : {eIMFs.shape}')
        print(f'residue shape : {residue.shape}')

        # 創建圖表
        fig, axs = plt.subplots(eIMFs.shape[0] + 2, 1, figsize=(10, 12))

        # 繪製原始訊號
        axs[0].plot(data, 'r')
        axs[0].set_title("Original Signal")

        # 繪製每個 IMF
        for i, imf in enumerate(eIMFs):
            axs[i+1].plot(imf)
            axs[i+1].set_title(f"IMF {i+1}")

        # 繪製殘差
        axs[-1].plot(residue)
        axs[-1].set_title("Residual")

        # 顯示圖表
        plt.tight_layout()
        plt.show()

        # 將殘差合併到 imfs 中
        residue_reshaped = residue.reshape(1, -1) 
        imfs = np.vstack((eIMFs, residue_reshaped))

        return eIMFs, residue, imfs
    

    def normalize(self, data_frame, labels, x_scaler = None, y_scaler = None):
        """最小最大正規化

        Args:
            data_frame (DataFrame): 數據集
            labels (array): 標籤欄位名稱陣列
            x_scaler (MinMaxScaler, optional): 特徵縮放器. Defaults to None. 若為 None 會執行 MinMaxScaler()
            y_scaler (MinMaxScaler, optional): 標籤縮放器. Defaults to None. 若為 None 會執行 MinMaxScaler()

        Returns:
            x_norm (array): 正規化後的特徵
            y_norm (array): 正規化後的標籤
            x_scaler (MinMaxScaler): 特徵縮放器
            y_scaler (MinMaxScaler): 標籤縮放器
            x_df (DataFrame): 正規化前的特徵 DataFrame
            y_df (DataFrame): 正規化前的標籤 DataFrame
        """
        print('\033[93m# 最小最大正規化\x1b[0m')

        if x_scaler == None:
            x_scaler = MinMaxScaler()
        if y_scaler == None:
            y_scaler = MinMaxScaler()

        x_df = data_frame
        y_df = data_frame[labels]

        x_norm = x_scaler.fit_transform(data_frame.values)
        y_norm = y_scaler.fit_transform(data_frame[labels].values)

        print(f'x_norm.shape:{x_norm.shape}')
        print(f'y_norm.shape:{y_norm.shape}')

        return x_norm, y_norm, x_scaler, y_scaler, x_df, y_df
 

    def normalize_by_decomposition(self, data, x_scaler = None, y_scaler = None):
        """最小最大正規化
        
        Args:
            data (array): 數據集
            x_scaler (MinMaxScaler, optional): 特徵縮放器. Defaults to None. 若為 None 會執行 MinMaxScaler()
            y_scaler (MinMaxScaler, optional): 標籤縮放器. Defaults to None. 若為 None 會執行 MinMaxScaler()
        
        Returns:
            x_norm (array): 正規化後的特徵
            y_norm (array): 正規化後的標籤
            x_scaler (MinMaxScaler): 特徵縮放器
            y_scaler (MinMaxScaler): 標籤縮放器
        """
        print('\033[93m# 最小最大正規化\x1b[0m')
        
        if x_scaler == None:
            x_scaler = MinMaxScaler()
        if y_scaler == None:
            y_scaler = MinMaxScaler()

        x_norm = x_scaler.fit_transform(data.reshape(-1, 1))
        y_norm = y_scaler.fit_transform(data.reshape(-1, 1))

        print(f'x_norm.shape:{x_norm.shape}')
        print(f'y_norm.shape:{y_norm.shape}')
        
        return x_norm, y_norm, x_scaler, y_scaler


    def train_val_test_split(self, data, train_split_rate, val_split_rate):
        """將資料集分割成訓練、驗證和測試集

        Args:
            data (array): 資料集
            train_split_rate (float): 訓練集佔整體資料集的比例
            val_split (float): 驗證集占整體資料集的比例

        Returns:
            x_train (array): 訓練集特徵
            y_train (array): 訓練集標籤
            x_val (array): 驗證集特徵
            y_val (array): 驗證集標籤
            x_test (array): 測試集特徵
            y_test (array): 測試集標籤
        """
        print('\033[93m# 切割訓練、驗證和測試資料集\x1b[0m')
 
        data_length = len(data)

        # 計算每個部分的大小
        train_size = int(data_length * train_split_rate)
        val_size = int(data_length * val_split_rate)
        test_size = data_length - train_size - val_size  # 剩下的就是測試集大小

        # 進行切割
        x_train = data[:train_size]
        y_train = data[:train_size]

        x_val = data[train_size:train_size + val_size]
        y_val = data[train_size:train_size + val_size]

        x_test = data[train_size + val_size:]
        y_test = data[train_size + val_size:]

        # 確認每個部分的大小
        print(f'x_train.shape: {x_train.shape}')
        print(f'y_train.shape: {y_train.shape}')
        print(f'x_val.shape: {x_val.shape}')
        print(f'y_val.shape: {y_val.shape}')
        print(f'x_test.shape: {x_test.shape}')
        print(f'y_test.shape: {y_test.shape}')

        return x_train, y_train, x_val, y_val, x_test, y_test


    def split_train_test_data(self, data, train_split_rate): 
        """將資料集分割成訓練和測試集

        Args:
            data (array): 資料集
            train_split_rate (float): 訓練集佔整體資料集的比例 

        Returns:
            x_train (array): 訓練集特徵
            y_train (array): 訓練集標籤 
            x_test (array): 測試集特徵
            y_test (array): 測試集標籤
        """
        print(f'\033[93m# 切割訓練和測試資料集，split rate {train_split_rate}\x1b[0m')
        
        data_length = len(data)
        
        # 計算要分割的資料點的索引
        split_index = int(data_length * train_split_rate)

        # 將資料分割為訓練集和測試集
        x_train, x_test = data[:split_index], data[split_index:]
        y_train, y_test = data[:split_index], data[split_index:]
 
        # 確認每個部分的大小
        print(f'x_train.shape: {x_train.shape}')
        print(f'y_train.shape: {y_train.shape}') 
        print(f'x_test.shape: {x_test.shape}')
        print(f'y_test.shape: {y_test.shape}')
         
        return x_train, y_train, x_test, y_test


    # def reshape_data_for_lstm(self, features, labels, date_array = None, past_day=14, future_day=1):
    #     """將資料轉成 LSTM 模型所需資料格式

    #     Args:
    #         features (array): 特徵數據
    #         labels (array): 標籤數據
    #         date_array (array): 標籤的日期，方便後續視覺化顯示預測日期用
    #         past_day (int, optional): 過去天數，用於訓練資料中的歷史資料長度. Defaults to 14.
    #         future_day (int, optional): 預測天數. Defaults to 1.

    #     Returns:
    #         x_lstm (array): 切割成 LSTM 模型所需的特徵
    #         y_lstm (array): 切割成 LSTM 模型所需的標籤
    #         date_lstm (array): 標籤的日期，方便後續視覺化顯示預測日期用
    #     """
    #     print('\033[93m# 將資料轉成 LSTM 模型所需資料格式，LSTM 的輸入必須是三維\x1b[0m')

    #     x_lstm, y_lstm, date_lstm = [], [], []

    #     index_day = 0 # 前幾天要包含當日 1，若不包含當日，則輸入 0
    #     # 4/4 日為例
    #     # ex. 含當日，4/2、4/3、4/4，預測 4/5
    #     # ex. 不含當日，4/2、4/3，預測 4/5

    #     for i in range(features.shape[0] - future_day - past_day + index_day):
    #         # 根據過去多少天的資料來構建輸入特徵
    #         x_lstm.append(np.array(features[i: i+past_day]))

    #         # 根據未來多少天的資料來構建輸出標籤
    #         y_lstm.append(np.array(labels[i+past_day: i+past_day+future_day]))

    #     if date_array is not None and date_array.size > 0:
    #         # 標籤的預設日，方便後續視覺化顯示預測日期用
    #         date_lstm.append(date_array[i+past_day])

    #     # # 將列表轉換為 dtype 為 object 的 ndarray
    #     # x_lstm = np.array(x_lstm, dtype=object)
    #     # y_lstm = np.array(y_lstm, dtype=object)

    #     return np.array(x_lstm), np.array(y_lstm), date_lstm
    
    
    def reshape_data_for_lstm(self, features, labels, date_array = None, past_day=14, future_day=1):
        """將資料轉成 LSTM 模型所需資料格式

        Args:
            features (array): 特徵數據
            labels (array): 標籤數據
            date_array (array): 標籤的日期，方便後續視覺化顯示預測日期用
            past_day (int, optional): 過去天數，用於訓練資料中的歷史資料長度. Defaults to 14.
            future_day (int, optional): 預測天數. Defaults to 1.

        Returns:
            x_lstm (array): 切割成 LSTM 模型所需的特徵
            y_lstm (array): 切割成 LSTM 模型所需的標籤
            date_lstm (array): 標籤的日期，方便後續視覺化顯示預測日期用
        """
        print('\033[93m# 將資料轉成 LSTM 模型所需資料格式，LSTM 的輸入必須是三維\x1b[0m')

        x_lstm, y_lstm, date_lstm = [], [], []
 
        # 上面寫法有 BUG 沒有真的抓到後天的資料，所以改寫如下
        # past_days = 14  # 使用過去 14 天作為輸入
        target_day = past_day + future_day + 1  # 預測第 16 天（不包含當天）
        # target_day = past_days + future_day  # 預測第 15 天（當天）
        
        # 迴圈遍歷資料，確保不包含當日，且正確預測未來的第 16 天
        for i in range(features.shape[0] - past_day - (target_day - past_day) + 1):
            # 取過去 1~14 天作為輸入
            input_sequence = features[i:i + past_day]
            x_lstm.append(input_sequence)
            
            target = np.array(labels[i + target_day - 1])
            y_lstm.append(target.reshape(1))
              
        if date_array is not None and date_array.size > 0:
            # 標籤的預設日，方便後續視覺化顯示預測日期用
            date_lstm.append(date_array[i + past_day]) 
        
        return np.array(x_lstm), np.array(y_lstm), date_lstm
    

    def build_model(self, model_type, time_step, n_predictions):
        """建立模型

        Args:
            model_type (str): 模型類型
            time_step (int): 時間步長
            n_predictions (int): 預測天數

        Returns:
            Sequential: 模型
        """
        print(f'\033[93m# 定義 Model：{model_type}\x1b[0m')
        # input_shape [Batch-size, TimeSteps, Features] [0, 14, 1]
        input_shape = [0, time_step, n_predictions]

        # 初始化模型
        model = Sequential()

        if model_type == 1: # LSTM
            print('LSTM')
            # https://www.sciencedirect.com/science/article/pii/S2352484722012719#fd16
            # 輸入層與第一個隱藏層，LSTM層使用100個神經元
            model.add(LSTM(100, input_shape=(input_shape[1], input_shape[2]), activation='tanh'))

            # 第二個隱藏層
            model.add(Dense(150, activation='tanh'))
            # model.add(Dropout(0.1))

            # 第三個隱藏層
            model.add(Dense(150, activation='tanh'))
            # model.add(Dropout(0.1))

            # 最後的輸出層，n_predictions 根據你的需求設置
            # 這裡假設n_predictions是你想要的輸出數量
            model.add(Dense(n_predictions))
        elif model_type == 2: # BiLSTM
            print('BiLSTM')
            model.add(Bidirectional(LSTM(100, activation='tanh'), input_shape=(input_shape[1], input_shape[2])))
            model.add(Dense(150, activation='tanh'))
            model.add(Dense(150, activation='tanh'))
            model.add(Dense(n_predictions))
        elif model_type == 3: # CNN-LSTM
            print('CNN-LSTM')
            # 添加一維卷積層
            model.add(Conv1D(filters=7, kernel_size=2, activation='relu', input_shape=(input_shape[1], input_shape[2])))
            model.add(MaxPooling1D(pool_size=2)) # 最大池化層

            model.add(LSTM(100, activation='tanh'))
            model.add(Dense(150, activation='tanh'))
            model.add(Dense(150, activation='tanh'))
            model.add(Dense(n_predictions))
        elif model_type == 4: # CNN-BiLSTM
            print('CNN-BiLSTM')
            model.add(Conv1D(filters=7, kernel_size=2, activation='relu', input_shape=(input_shape[1], input_shape[2])))
            model.add(MaxPooling1D(pool_size=2))

            model.add(Bidirectional(LSTM(100, activation='tanh')))
            model.add(Dense(150, activation='tanh'))
            model.add(Dense(150, activation='tanh'))
            model.add(Dense(n_predictions))

        # 定義訓練方式，ex. 使用均方誤差（mean squared error）作為損失函數，Adam 優化器進行優化
        # model_optimizer = 'adam'  # 模型優化器，例如 "adam", "sgd" 等
        # 參考參數 : https://datascience.stackexchange.com/questions/48736/how-to-see-change-learning-rate-in-keras-lstm
        model_optimizer = Adam(learning_rate=0.001) # Defaults to 0.001.

        model.compile(
            metrics = self.model_metrics,
            optimizer = model_optimizer,
            loss = self.model_loss
        )

        model.summary()

        return model

    def model_fit(self, model, batch_size, x_train, y_train, x_val, y_val):
        """訓練模型

        Args:
            model (Sequential): 模型
            x_train (array): 訓練集特徵
            y_train (array): 訓練集標籤
            x_val (array): 驗證集特徵
            y_val (array): 驗證集標籤

        Returns:
            History: 訓練歷史
        """
        print('\033[93m# 訓練 Model\x1b[0m')

        earlystop_monitor_loss = 'val_loss'  # 提早停止監控的損失 loss or val_loss
        earlystop_epochs = 50  # 提早停止迭代次數
        earlystop = EarlyStopping(
          monitor = earlystop_monitor_loss,    # 監控的指標，這裡是設置為監控模型的損失值
          patience = earlystop_epochs,       # 當被監控指標停止改善時，經過多少個epoch後停止訓練
          mode = "min"         # 監控模式，這裡是設置為最小化模式，即監控指標越小越好
        )
        print(f'EarlyStopping => mointor: {earlystop_monitor_loss}，patience: {earlystop_epochs}')

        # 載入上次訓練的權重
        train_history = model.fit(
            x = x_train,
            y = y_train,
            validation_data=(x_val, y_val),

            batch_size = batch_size, # 將訓練的資料分成多批
            epochs = self.epochs, # 每一個 epoch 包含了 n 批的訓練 + 1 次驗證
            verbose = 0, # 0:不顯示任何訊息、1: 顯示完整訊息、2: 只顯示訓練驗證資訊
            callbacks=[earlystop]
        )

        return train_history


    def show_train_history_chart(self, train_history, title = ''):
        """顯示訓練歷史圖表

        Args:
            train_history (History): 訓練歷史
            title (str): 圖表標題
        """
        plt.figure(figsize=(6, 3))

        # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        train_loss = train_history.history['loss']

        print('最後一個 Epoch Train Loss:', train_loss[-1])

        valid_key = 'val_loss'
        plt.plot(train_history.history['loss'])
        if valid_key in train_history.history:
            plt.plot(train_history.history[valid_key], '--')
        plt.title(f'History{title}')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        if valid_key in train_history.history:
            plt.legend(['train', 'validation'], loc='upper right')
        else:
            plt.legend(['train'], loc='upper right')

        plt.show()


    def predict(self, model, x):
        """預測

        Args:
            model (Sequential): 模型
            x (array): 輸入特徵

        Returns:
            array: 預測結果
        """
        print('\033[93m# 預測\x1b[0m')
        predict = model.predict(x)
        print(f'預測 shape : {predict.shape}')

        return predict


    def denormalize_and_reshape_back(self, y_predict, y_lstm_actual, y_scaler, n_predictions):
        """還原預測資料值為原始數據的規模

        Args:
            y_predict (array): 預測結果
            y_lstm_actual (array): 實際值
            y_scaler (MinMaxScaler): 標籤縮放器
            n_predictions (int): 預測天數

        Returns:
            array: 還原後的預測結果
            array: 還原後的實際值
            array: 預測結果
            array: 實際值
        """

        # 回復預測資料值為原始數據的規模
        print('\033[93m# 回復預測資料值為原始數據的規模\x1b[0m')

        # 將預測值進行逆轉換
        predict_inverse = y_scaler.inverse_transform(y_predict)

        # 還原實際資料還原回原本的結構
        actual = y_lstm_actual.reshape(-1, n_predictions)
        actual_inverse = y_scaler.inverse_transform(actual) 

        return predict_inverse, actual_inverse, y_predict, actual
 


    def evaluation(self, model, x_lstm, y_lstm):
        """評估模型

        Args:
            model (Sequential): 模型
            x_lstm (array): 輸入特徵
            y_lstm (array): 實際值

        Returns:
            array: 評估結果
        """
        # 將 NumPy ndarray 轉換為 TensorFlow Tensor
        x_lstm_tensor = tf.convert_to_tensor(x_lstm, dtype=tf.float32)
        y_lstm_tensor = tf.convert_to_tensor(y_lstm, dtype=tf.float32)

        # 評估模型
        evaluation = model.evaluate(x=x_lstm_tensor, y=y_lstm_tensor)

        return evaluation


    def score(self, actual, predict):
        """計算評估指標

        Args:
            actual (array): 實際值
            predict (array): 預測值

        Returns:
            float: MAE
            float: MAPE
        """
        print(f'Actual shape: {actual.shape}, Predict shape: {predict.shape}')
        if actual.shape != predict.shape:
            raise ValueError("Shape mismatch between 'actual' and 'predict'")

        mae = mean_absolute_error(actual, predict)
        mape = mean_absolute_percentage_error(actual, predict)

        # squared True returns MSE value, False returns RMSE value.
        mse = mean_squared_error(actual, predict)
        rmse = mean_squared_error(actual, predict, squared=False)

        r_square = r2_score(actual, predict)

        result = f'MAE:{round(mae, 5)}，MAPE:{round(mape, 5)}，'\
            f'MSE:{round(mse, 5)}，RMSE:{round(rmse, 5)}，'\
            f'R square:{round(r_square, 5)}'

        print(result)

        return mae, mape


    def show_predict_chart(self, actual, predict, title):
        """顯示預測圖表

        Args:
            actual (array): 實際值
            predict (array): 預測值
            title (str): 圖表標題
        """
        plt.figure(figsize=(8, 2))
        plt.plot(actual)
        plt.show()

        plt.figure(figsize=(8, 2))
        plt.plot(actual, label='Actual')
        plt.plot(predict, label='Predicted')
        plt.title(title)

        plt.legend()
        plt.show()



    def save_json_file(
            self,
            result_file_name, type, labels, model_type, predicteds, 
            batch_size, mae_train, mape_train, mae_test, mape_test,
            kfold = None,
            imfs = None
    ):
        """儲存評估結果到 JSON 檔案

        Args:
            result_file_name (str): 結果檔案名稱
            type (str): 模型類型
            labels (str): 標籤
            model_type (int): 模型類型
            predicteds (str): 預設類別
            batch_size (int): 批次大小
            mae_train (float): 訓練 MAE
            mape_train (float): 訓練 MAPE
            mae_test (float): 測試 MAE
            mape_test (float): 測試 MAPE
            kfold (int, optional): K-Fold 交叉驗證. Defaults to None.
            imfs (array, optional): 分解後的 IMFs. Defaults to None.
        """
        type = type.replace('/_Model', '')  
    
        json_result = {}
        json_result["type"] = type
        json_result["labels"] = labels
        json_result["model_type"] = model_type
        json_result["predicteds"] = predicteds
        json_result["batch_size"] = batch_size
        json_result["train_mae"] = round(mae_train, 5)
        json_result["train_mape"] = round(mape_train, 5)
        json_result["test_mae"] = round(mae_test, 5)
        json_result["test_mape"] = round(mape_test, 5)
        json_result["kfold"] = kfold
        json_result["imfs"] = imfs

        print(json_result)
        with open(result_file_name, 'w') as json_file:
            json.dump(json_result, json_file)     
            print(f'儲存評估結果到 {result_file_name}')