import pandas as pd
import matplotlib.pyplot as plt 
# pip install sqlalchemy pyodbc
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text

class Database:  
    def __init__(self, server, database, username, password): 
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  
        
        password = quote_plus(password)
        
        self.connection_string = f"mssql+pyodbc://{username}:{password}@{server}/{database}?Encrypt=no&driver=ODBC+Driver+18+for+SQL+Server"
        self.engine = create_engine(self.connection_string)
        
    def insert_to_predictions(self, df): 
        """
        將預測結果寫入資料庫
        
        Args:
            df (pandas.DataFrame): 資料集
        """
        
        # 將 DataFrame 轉換為 SQL VALUES 格式
        values = ", ".join(
            f"('{row.RequestTime}', '{row.Type}', {row.SendBytes}, {row.ReceiveBytes}, {row.TimeTaken}, {row.GroupCount})"
            for row in df.itertuples(index=False)
        )
        
        sql = """
            MERGE INTO IT_IISLog_Predictions AS target
            USING (VALUES {values})
            AS source (RequestTime, Type, SendBytes, ReceiveBytes, TimeTaken, GroupCount)
            ON (target.RequestTime = source.RequestTime AND target.Type = source.Type)
            WHEN MATCHED THEN 
                UPDATE SET 
                    target.SendBytes = source.SendBytes,
                    target.ReceiveBytes = source.ReceiveBytes,
                    target.TimeTaken = source.TimeTaken,
                    target.GroupCount = source.GroupCount,
                    target.ModifiedDate = GETUTCDATE()
            WHEN NOT MATCHED THEN 
                INSERT (RequestTime, Type, SendBytes, ReceiveBytes, TimeTaken, GroupCount)
                VALUES (source.RequestTime, source.Type, source.SendBytes, source.ReceiveBytes, source.TimeTaken, source.GroupCount);
        """
 
        final_query = sql.format(values=values)
        # print(final_query)
        
        with self.engine.connect() as connection:
            with connection.begin(): 
                result = connection.execute(text(final_query))
                print(f'\t{result.rowcount} 筆資料受影響')
                
    
    def query_data(self, sdate, edate, type): 
        """
        查詢資料
        
        Args:
            sdate (str): 起始日期
            edate (str): 結束日期
            type (str): 類型
            
        Returns:
            pandas.DataFrame: 資料集
        """
        sql = """
            Select RequestTime, Type, SendBytes, ReceiveBytes, TimeTaken, GroupCount
            From IT_IISLog_Predictions
            Where RequestTime Between :sdate and :edate and Type = :type
            Order by RequestTime
        """
        query_param = {
            'sdate': sdate,
            'edate': edate,
            'type': type
        }
        
        with self.engine.connect() as connection: 
            query = text(sql)
            df = pd.read_sql(query, connection, params=query_param)
    
            return df
            
            
    def query_top_n_data(self, date, n, type): 
        """
        查詢指定日期前的 N 筆資料
        
        Args:   
            date (str): 日期
            n (int): 筆數
            type (str): 類型
            
        Returns:
            pandas.DataFrame: 資料集
        """
        sql = f"""
            Select * 
            From
            ( 
                Select RequestTime, Type, SendBytes, ReceiveBytes, TimeTaken, GroupCount, 
                    GroupCountPredicted, GroupCountPredicted2
                From IT_IISLog_Predictions
                Where RequestTime <= :date and Type = :type 
                -- and (GroupCount is not null and GroupCount > 0)
                Order by RequestTime desc
                OFFSET 0 ROWS FETCH NEXT {n} ROWS ONLY
            ) main 
            Order By RequestTime ASC
        """
        query_param = {
            'date': date, 
            'type': type
        }
        
        with self.engine.connect() as connection: 
            query = text(sql)
            df = pd.read_sql(query, connection, params=query_param)
    
            return df
        
        
    def fill_groupcount_with_median(self, df: pd.DataFrame, label_name: str = 'GroupCount', days_range: int = 30):
        """
        將 指定欄位 為 null 或 0 的資料，取指定日期(含)30天內的中位數填補
        
        Args:
            df (pandas.DataFrame): 資料集
            label_name (str): 欄位名稱
            days_range (int): 日期範圍
            
        Returns:
            pandas.DataFrame: 資料集
        """
        date_column = 'RequestTime'
        
        # 確保日期欄位是 pandas datetime 類型
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        # 篩選出 label 為 null 或 0 的資料
        mask_null_or_zero = df[label_name].isnull() | (df[label_name] == 0)

        # 逐一處理每一行，填補 null 或 0 的值
        for idx, row in df[mask_null_or_zero].iterrows():
            current_date = row[date_column]
            start_date = current_date - pd.Timedelta(days=days_range)  # 計算30天前的日期
            end_date = current_date  # 當前日期作為範圍的結束
            
            # 找到30天內的資料
            data_in_range = df[(df[date_column] >= start_date) & (df[date_column] < end_date)]
            
            # 計算30天內的中位數
            median_value = data_in_range[label_name].median()
            
            # 用中位數替代 null 或 0 的值
            if pd.notna(median_value):  # 確保中位數有效（避免中位數為 NaN）
                df.at[idx, label_name] = median_value

        return df
    
    
    def insert_to_traffic_predictions(self, type, label_name, predicted_label_name, date_data, prediction_data):
        """
        將預測結果寫入資料庫
        
        Args: 
            type (str): 類型
            label_name (str): 標籤名稱
            predicted_label_name (str): 預測標籤名稱
            date_data (list): 日期資料
            prediction_data (list): 預測資料
        """
         
        # 將 ndarray 轉換為 SQL VALUES 格式   
        values = ", ".join(
            f"('{date.strftime('%Y-%m-%d')}', '{type}', {predicted.round()})"
            for date, predicted in zip(date_data.reshape(-1), prediction_data.reshape(-1))
        )
        
        sql = f"""
            MERGE INTO IT_IISLog_Predictions AS target
            USING (VALUES {values})
            AS source (RequestTime, Type, {predicted_label_name})
            ON (target.RequestTime = source.RequestTime AND target.Type = source.Type)
            WHEN MATCHED THEN 
                UPDATE SET  
                    target.{predicted_label_name} = source.{predicted_label_name},
                    target.ModifiedDate = GETUTCDATE()
            WHEN NOT MATCHED THEN 
                INSERT (RequestTime, Type, {predicted_label_name})
                VALUES (source.RequestTime, source.Type, source.{predicted_label_name});
        """
 
        final_query = sql.format(values=values)
        # print(final_query)
        
        with self.engine.connect() as connection:
            with connection.begin(): 
                result = connection.execute(text(final_query))
                print(f'\t{result.rowcount} 筆資料受影響')
                
                
    def get_threadhold(self, type, is_vmd = False): 
        """
        取得閾值
        
        Args:
            type (str): 類型
            
        Returns:
            float: 閾值上限
            float: 閾值下限
        """ 
        threshold_up = 0
        threshold_down = 0

        if is_vmd == False: # 使用 LSTM2 NoDecomp 測試集的 MAE 值當做閾值
            if type == '1':
                threshold_up = 17807 
                threshold_down = threshold_up * 0.5
            elif type == '246':
                threshold_up = 557 * 2
                threshold_down = 557 * 3
            elif type == '3':
                threshold_up = 8281 * 2
                threshold_down = threshold_up
            elif type == '5':
                threshold_up = 687 * 1.5
                threshold_down = 687 * 2.5
        else:  # 使用整批預測的 MAE 值當做閾值 LSTM4 VMD_NoDecomp 2023-04-02 ~ 2024-04-30
            if type == '1':  
                threshold_up = 6397 * 2
                threshold_down = threshold_up 
            elif type == '246': 
                threshold_up = 970 * 2
                threshold_down = threshold_up
            elif type == '3':
                threshold_up = 6824 * 2
                threshold_down = threshold_up
            elif type == '5':
                threshold_up = 819 * 2
                threshold_down = threshold_up
                
        return threshold_up, threshold_down    
            
            
    def show_predict_chart(self, actual_data, prediction_data, threshold_up, threshold_down, label_name, date_data = None):
        """
        顯示預測圖表
        
        Args:
            actual_data (numpy.ndarray): 實際資料
            prediction_data (numpy.ndarray): 預測資料
            threshold_up (float): 閾值上限
            threshold_down (float): 閾值下限
            label_name (str): 標籤名稱
            date_data (numpy.ndarray): 日期資料
            
        Returns:
            None
        """
        
        y_data = actual_data.reshape(-1)  
        
        plt.figure(figsize=(10, 3))
        ax = plt.gca()  # Get current axis
        self.plot_threshold_and_anomalies(ax, date_data, y_data, prediction_data, threshold_up, threshold_down, label_name)
        
        plt.show() 
        
        
    def plot_threshold_and_anomalies(self, ax, date_data, y_data, 
        prediction_data, threshold_up, threshold_down, label_name, verbose = False
    ):
        # 每一個預測值 - 閾值下邊界 & 下限不得低於 0
        threshold_upper = [p + threshold_up for p in prediction_data]
        threshold_upper = [0 if p < 0 else p for p in threshold_upper]
        
        threshold_lower = [p - threshold_down for p in prediction_data]
        threshold_lower = [0 if p < 0 else p for p in threshold_lower]
        
        # 實際資料
        ax.plot(date_data, y_data, label='實際(原始)', marker='o', markersize=1) 
        
        # 使用 fill_between 填充閾值區域為淡綠色
        ax.fill_between(date_data, threshold_lower, threshold_upper, color='green', alpha=0.2, label='閾值區域')

        # 異常檢測，並用紅點標示異常值
        anomalies = []
        for i in range(len(y_data)):
            if y_data[i] > threshold_upper[i] or y_data[i] < threshold_lower[i]:
                # if y_data[i] - 0 < threshold_down:  # Skip small anomalies
                #     continue
                anomalies.append((i, date_data[i], y_data[i]))
                ax.scatter(date_data[i], y_data[i], color='red', s=15) 
        
        if anomalies:
            anomaly_range = f'異常點數量: {len(anomalies)}'
            ax.scatter([], [], color='red', s=15, label=anomaly_range)
        else:
            ax.scatter([], [], color='red', s=15, label='無異常點')
        
        if verbose:
            print(f'異常數量: {len(anomalies)}')
            # print(f'異常值: {anomalies}')
            
            df = pd.DataFrame({
                'IsAnomaly': [y_data[i] > threshold_upper[i] or y_data[i] < threshold_lower[i] for i in range(len(y_data))],
                'RequestTime': date_data, 
                'Actual': y_data, 
                'Prediction': prediction_data,
                'threshold_upper': threshold_upper,
                'threshold_lower': threshold_lower
            }) 
            # print(df.to_string(index=False))
            # 僅顯示異常值
            print(df[df['IsAnomaly'] == True].to_string(index=False))
            print(len(df[df['IsAnomaly'] == True]))
        
        ax.set_title(f'{label_name}')
        ax.legend()


    def show_predict_chart2(self, actual_data, type,
        prediction_data, threshold_up, threshold_down, 
        prediction_data2, threshold_up2, threshold_down2, 
        label_name, 
        date_data = None
    ):
        
        y_data = actual_data.reshape(-1)
        
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)  # Create two subplots

        # Plot for prediction_data and threshold (sub-plot 1) 
        ax1 = axes[0]
        self.plot_threshold_and_anomalies(ax1, date_data, y_data, prediction_data, threshold_up, threshold_down, 
            f'{label_name}({type}) - 預測(無異常處理)'
        )
        
        # Plot for prediction_data2 and threshold (sub-plot 2)  
        ax2 = axes[1]
        self.plot_threshold_and_anomalies(ax2, date_data, y_data, prediction_data2, threshold_up2, threshold_down2, 
            f'{label_name}({type}) - 預測(異常處理)'
        )

        # prediction_data and prediction_data2 平均值
        prediction_data_avg = (prediction_data + prediction_data2) / 2
        
        # Plot for prediction_data_avg and threshold (sub-plot 3)
        ax3 = axes[2]
        self.plot_threshold_and_anomalies(ax3, date_data, y_data, prediction_data_avg, threshold_up, threshold_down, 
            f'{label_name}({type}) - 預測(使用閾值1平均)'
        )
        
        # Plot for prediction_data_avg and threshold (sub-plot 4)
        ax4 = axes[3]
        self.plot_threshold_and_anomalies(ax4, date_data, y_data, prediction_data_avg, threshold_up2, threshold_down2, 
            f'{label_name}({type}) - 預測(使用閾值2平均)', verbose = True
        )
           
        plt.tight_layout()
        # plt.savefig(f'{label_name}({type}).png')
        plt.show()
