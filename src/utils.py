import csv 

# In fact, we use the pd.datetime 
import datetime 
import time 

import pandas as pd 
import numpy as np 
import scipy as sp 


def get_company_list(n=500):
    temp = pd.read_csv('../SP500.csv')
    return np.array(temp['Symbol'])[:n]
        
def get_esg_list(k=100):
    company_list=get_company_list(k)
    
    esg_list=np.unique(np.array(pd.read_csv('../data/esg/{}_esg.csv'.format(company_list[0]))['aspectname']))
    
    for i in company_list: 
        try : 
            temp_list = np.unique(np.array(pd.read_csv('../data/esg/{}_esg.csv'.format(i))['aspectname']))
            esg_list = np.intersect1d(temp_list, esg_list)
        except:
            pass
    return esg_list



class company():
    # Timeformat, pd datetime, convenient to use with the datetime array. 
    start = pd.to_datetime('1970-01-01',format='%Y-%m-%d')
    end = pd.to_datetime('2023-11-21',format='%Y-%m-%d')
    frequency = pd.Timedelta(days=1) 
    
    # We need the ticker name of the company to get started 
    def __init__(self,ticker,autofill=True):
        self.ticker = ticker  
        self.constrain_time_range()
        self.generate_time_series()

    def __repr__(self):
        return self.ticker
    def __str__(self):
        return "Company Class :{}".format(self.ticker)
 
    # Start "%Y-%m-%d" , End "%Y-%m-%d" , Frequency N days. 
    def update_time_range(self,start,end,frequency) : 
        self.start = pd.to_datetime(start,format='%Y-%m-%d')
        self.end = pd.to_datetime(end,format='%Y-%m-%d')
        self.frequency = pd.Timedelta(days=frequency)
        self.constrain_time_range()
        self.generate_time_series()

        
    def constrain_time_range(self):
        # The temp is to save the readout data 
        temp = pd.read_csv('../data/price/{}_price.csv'.format(self.ticker))
        
        # Filter the data within the interesting time  area. 
        referdate = pd.to_datetime( temp['Date'], format='%Y-%m-%d') 
        
        price_start = np.min(np.array(referdate)) 
        price_end = np.max(np.array(referdate)) 
        
        if self.start < price_start : 
            self.start = price_start 
        
        if self.end > price_end : 
            self.end = price_end 
        
        return 
        
    def generate_time_series(self): 
        data_points_number = int((self.end - self.start)/(self.frequency)) +1 
        self.time_series = np.array([ x*self.frequency + self.start for x in range(data_points_number) ])
    
    
    
    
    # To verify the integrity of the data .... 
    def feature_list(self): 
        pass 
    
    def print_feature_list(self):
        pass 
    
    def check_data(self):
        pass 


    def read_data(self,target_list):
        query_result = np.zeros((len(target_list), len(self.time_series) ))
        if len(target_list) == 1 : 
            # the 2d numpy array (1,n) behaves quite strange 
            query_result = self.read_single_data(target_list[0])
        else :
            for i in range(len(target_list)):
                query_result[i]= self.read_single_data(target_list[i])
        
        return (self.time_series,query_result)
    
    def read_single_data(self,target):
        if target[0]=='price':
            return self.read_single_price_data(target)
        elif target[0]=='balance' : 
            return self.read_single_balance_data(target)
        elif target[0]=='cashflow' : 
            return self.read_single_cashflow_data(target)
        elif target[0]=='esg' : 
            return self.read_single_esg_data(target)
        elif target[0]=='financials':
            return self.read_single_financials_data(target)
        elif target[0]=='valuation':
            return self.read_single_valuation_data(target)
        else : 
            print("Warning, make sure that you type the right feature name")
    
    def read_single_price_data(self,target):
        # The temp is to save the readout data 
        temp = pd.read_csv('../data/price/{}_price.csv'.format(self.ticker))
        
        # Filter the data within the interesting time  area. 
        referdate = pd.to_datetime( temp['Date'], format='%Y-%m-%d') 
        temp_date = np.array(referdate[ np.array( (referdate <= self.end) &  (referdate >= self.start) )])
        temp_data = np.array(temp[target[1]][ np.array( (referdate <= self.end) &  (referdate >= self.start) )])
        
        # reorganzie the data sequence 
        date_arg = np.array(np.argsort(temp_date))
        temp_date = temp_date[date_arg]
        temp_data = temp_data[date_arg]
        
        # Not Every data works, and then we will rebuild the data according to the the time series 
        # For the timebeing, we will use the uniform choice method 
        return_args = np.array( np.arange(len(self.time_series))* len(temp_date) / len(self.time_series) ).astype(int)
        return_data = np.array(temp_data[return_args])
        
        return return_data

    
    def read_single_balance_data(self,target):
        temp = pd.read_csv('../data/balance/{}_balance.csv'.format(self.ticker))
        
        row_name = temp['name']
        n_row = len(row_name)
        target_row = 0 
        
        # find the columns we want 
        for i in range(n_row):
            if row_name[i].strip() == target[1] : 
                target_row = i 
                break 
        
        # Filter the data within the 
        referdate = pd.to_datetime( np.array(temp.columns[1:]) ,format='%m/%d/%Y' )
        temp_date= np.array(referdate[ np.array( (referdate <= self.end) &  (referdate >= self.start) )])
        temp_data= np.array(temp.iloc[i][1:][ np.array( (referdate <= self.end) &  (referdate >= self.start) )])
        
        # reorganzie the data sequence 
        date_arg = np.argsort(temp_date)
        temp_date = temp_date[date_arg]
        temp_data = temp_data[date_arg]
        
        # Not Every data works, and then we will rebuild the data according to the the time series 
        # For the timebeing, we will use the uniform choice method 
        return_args = np.array( np.arange(len(self.time_series))* len(temp_date) / len(self.time_series) ).astype(int)
        return_data = np.array(temp_data[return_args])
        return_data = [ float(i.replace(',','')) for i in return_data ]
        return return_data
    
    def read_single_cashflow_data(self,target):
        temp = pd.read_csv('../data/cashflow/{}_cashflow.csv'.format(self.ticker))
        
        row_name = temp['name']
        n_row = len(row_name)
        target_row = 0 
        
        # find the columns we want 
        for i in range(n_row):
            if row_name[i].strip() == target[1] : 
                target_row = i 
                break 
        
        # Filter the data within the 
        referdate = pd.to_datetime( np.array(temp.columns[2:]) ,format='%m/%d/%Y' )
        temp_date= np.array(referdate[ np.array( (referdate <= self.end) &  (referdate >= self.start) )])
        temp_data= np.array(temp.iloc[i][2:][ np.array( (referdate <= self.end) &  (referdate >= self.start) )])

        # reorganzie the data sequence 
        date_arg = np.argsort(temp_date)
        temp_date = temp_date[date_arg]
        temp_data = temp_data[date_arg]        
        
        # Not Every data works, and then we will rebuild the data according to the the time series 
        # For the timebeing, we will use the uniform choice method 
        return_args = np.array( np.arange(len(self.time_series))* len(temp_date) / len(self.time_series) ).astype(int)
        return_data = np.array(temp_data[return_args])     
        return_data = [ float(i.replace(',','')) for i in return_data ]
        return return_data 
 
    
    def read_single_financials_data(self,target):
        temp = pd.read_csv('../data/financials/{}_financials.csv'.format(self.ticker))
        
        row_name = temp['name']
        n_row = len(row_name)
        target_row = 0 
        
        # find the columns we want 
        for i in range(n_row):
            if row_name[i].strip() == target[1] : 
                target_row = i 
                break 
        
        # Filter the data within the 
        referdate = pd.to_datetime( np.array(temp.columns[2:]) ,format='%m/%d/%Y' )
        temp_date= np.array(referdate[ np.array( (referdate <= self.end) &  (referdate >= self.start) )])
        temp_data= np.array(temp.iloc[i][2:][ np.array( (referdate <= self.end) &  (referdate >= self.start) )])

        # reorganzie the data sequence 
        date_arg = np.argsort(temp_date)
        temp_date = temp_date[date_arg]
        temp_data = temp_data[date_arg]        
        
        # Not Every data works, and then we will rebuild the data according to the the time series 
        # For the timebeing, we will use the uniform choice method 
        return_args = np.array( np.arange(len(self.time_series))* len(temp_date) / len(self.time_series) ).astype(int)
        return_data = np.array(temp_data[return_args])     
        return_data = [ float(i.replace(',','')) for i in return_data ]
        return return_data  
    
    def read_single_valuation_data(self,target):
        temp = pd.read_csv('../data/valuation/{}_valuation.csv'.format(self.ticker))
        
        row_name = temp['name']
        n_row = len(row_name)
        target_row = 0 
        
        # find the columns we want 
        for i in range(n_row):
            if row_name[i].strip() == target[1] : 
                target_row = i 
                break 
        
        # Filter the data within the time range 
        referdate = pd.to_datetime( np.array(temp.columns[2:]) ,format='%m/%d/%Y' )
        temp_date= np.array(referdate[ np.array( (referdate <= self.end) &  (referdate >= self.start) )])
        temp_data= np.array(temp.iloc[i][2:][ np.array( (referdate <= self.end) &  (referdate >= self.start) )])

        # reorganzie the data sequence 
        date_arg = np.argsort(temp_date)
        temp_date = temp_date[date_arg]
        temp_data = temp_data[date_arg]        
        
        # Not Every data works, and then we will rebuild the data according to the the time series 
        # For the timebeing, we will use the uniform choice method 
        return_args = np.array( np.arange(len(self.time_series))* len(temp_date) / len(self.time_series) ).astype(int)
        return_data = np.array(temp_data[return_args])     
        return_data = [ float(i.replace(',','')) for i in return_data ]
        return return_data  

    
    
    def read_single_esg_data(self,target):
        temp = pd.read_csv('../data/esg/{}_esg.csv'.format(self.ticker))        
        target_row = np.where(np.array(temp['aspectname'])== target[1])[0]
        
        # Filter data 
        referdate = pd.to_datetime( np.array(temp.iloc[target_row]['scoredate']) ,format='%Y-%m-%d' )
        temp_date= np.array(referdate[ np.array( (referdate <= self.end) &  (referdate >= self.start) )])
        temp_data= np.array(temp.iloc[target_row]['scorevalue'][ np.array( (referdate <= self.end) &  (referdate >= self.start) )])

        # reorganzie the data sequence 
        date_arg = np.argsort(temp_date)
        temp_date = temp_date[date_arg]
        temp_data = temp_data[date_arg]        
        
        # Not Every data works, and then we will rebuild the data according to the the time series 
        # For the timebeing, we will use the uniform choice method 
        return_args = np.array( np.arange(len(self.time_series))* len(temp_date) / len(self.time_series) ).astype(int)
        return_data = np.array(temp_data[return_args])     
        return return_data 
    
    def plot_single_data(self,target):
        pass 