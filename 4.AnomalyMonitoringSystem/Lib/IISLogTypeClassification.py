import pandas as pd
import joblib 
import re

class IISLogTypeClassification:
    def __init__(self, model_dir): 
        self.model_dir = model_dir
        
    def load_datasource(self, file_path, yyyymmdd): 
        """
        讀取 IIS Log 的 csv 檔案到 pandas DataFrame
        
        Args:
            file_path (str): CSV 檔案路徑
            
        Returns:
            pandas.DataFrame: 資料集
        """
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        df = self.__uri_discrete(df)
        
        # 過濾出指定日期的資料 
        df = df[df['RequestTime'].str.contains(yyyymmdd)]
        
        return df


    def load_model(self):
        """
        載入模型
        
        Returns:
            object: 模型
        """
        model = joblib.load(f'{self.model_dir}/decision_tree_model.joblib') 
        return model 
     
     
    def convert_features(self, data_frame):
        """
        轉換預測用的特徵

        Args:
            data_frame (pandas.DataFrame): 資料集
            
        Returns:
            numpy.ndarray: 特徵
        """
        col_names = ['Status', 'SendBytes', 'ReceiveBytes', 'UrlType']
        
        # 刪除沒有用到分類特徵
        return data_frame[col_names].values
     
    
    def predict(self, model, features):
        """
        預測
        
        Args:
            model (object): 模型
            features (numpy.ndarray): 特徵
        
        Returns:
            numpy.ndarray: 預測結果
        """
        y_pred = model.predict(features)
        return y_pred   
    
    
    def append_predicted(self, data_frame, y_pred):
        """
        將預測結果加入原始資料集
        
        Args:
            data_frame (pandas.DataFrame): 資料集
            y_pred (numpy.ndarray): 預測結果
            
        Returns:
            pandas.DataFrame: 資料集
        """
        level2_fileds = ["RequestTime","FormattedRequestTime","Method","Status","UrlType","SendBytes","ReceiveBytes","TimeTaken","Predicted"]
    
        data_frame['RequestTime'] = pd.to_datetime(data_frame['RequestTime'])
        data_frame["Predicted"] = y_pred 
        data_frame['FormattedRequestTime'] = data_frame['RequestTime'].dt.floor('D')  # 格式化每一天一群
        data_frame = data_frame[level2_fileds]
        
        return data_frame
      
    def group_by_day(self, data_frame):
        """
        加總每天的數據
        
        Args:
            data_frame (pandas.DataFrame): 資料集
            
        Returns:
            pandas.DataFrame: 資料集
        """
        
        group_by_fields = ['FormattedRequestTime', 'Predicted'] # 分組欄位
        group_count_field_name = 'GroupCount' # 加總欄位名稱
    
        # 使用 groupby 進行分組，並且對其他欄位進行加總
        result_df = data_frame.groupby(group_by_fields).agg({
            'SendBytes': 'sum',
            'ReceiveBytes': 'sum',
            'TimeTaken': 'sum'
        }).reset_index()
         
        # 加總 group by 筆數
        result_df[group_count_field_name] = data_frame.groupby(group_by_fields).size()\
            .reset_index(name=group_count_field_name)[group_count_field_name]
  
        # 針對相同日期且 Predicted 是 2、4、6 的資料進行加總
        summary_df = result_df[result_df['Predicted'].isin([2, 4, 6])] \
            .groupby('FormattedRequestTime').agg({
                'SendBytes': 'sum',
                'ReceiveBytes': 'sum',
                'TimeTaken': 'sum',
                group_count_field_name: 'sum'
            }).reset_index()

        # 新增標記欄位
        summary_df['Predicted'] = '246' 

        # 合併加總結果到 result_df
        result_df = pd.concat([result_df, summary_df], ignore_index=True)
        
        # 根據 FormattedRequestTime 和 Predicted 欄位進行排序
        result_df = result_df.sort_values(by=['FormattedRequestTime', 'Predicted']).reset_index(drop=True)
    
        # 將欄位名稱 FormattedRequestTime 重命名為 RequestTime
        result_df = result_df.rename(columns={
            'FormattedRequestTime': 'RequestTime',
            'Predicted': 'Type'
        })
    
        return result_df 
    
    
    def __uri_discrete(self, data_frame): 
        """
        將 URI 進行離散化
        
        Args:
            dataFrame (pandas.DataFrame): 資料集
            
        Returns:
            pandas.DataFrame: 資料集
        """
        field_url_type = 'UrlType'
        # field_uri = 'URIStem'
        field_uri = 'URI'
  
        data_frame[field_url_type] = data_frame[field_uri].apply(self.__assign_uri_type) 
        
        # 移除 field_uri 欄位   
        data_frame = data_frame.drop(field_uri, axis=1)

        return data_frame
    
    def __assign_uri_type(self, uri):
        """
        決定 URI 離散化的類別
        
        Args:   
            uri (str): URI
            
        Returns:
            int: 類別
        """ 
        if uri.startswith('/OAuth/') or uri.startswith('/api/'):
            return 1 # 'api'
        elif uri == '/' or \
            uri == '/school' or \
            uri == '/about' or \
            uri == '/charge-map' or \
            uri == '/school-map' or \
            uri == '/punish-map' or \
            uri == '/.well-known/traffic-advice' or \
            uri.startswith('/school/') or \
            uri.startswith('/assets/') or \
            re.search(r'\.(html|css|js|jpg|jpeg|png|webp|ico|svg|gif)$', uri):
            return 2 #'static'
        else:
            return 3 #'other'