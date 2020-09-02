from flask import Flask
from flask import request
import time
from datetime import datetime
from pytz import timezone
import requests
import itertools
import re
import numpy as np
import pandas as pd
import sys, os

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from pandas.tseries.offsets import DateOffset

from influxdb import InfluxDBClient

import matplotlib.pyplot as plt
#%matplotlib inline
'exec(%matplotlib inline)'

error_message_url_format = "URL is not in correct format"
error_message_time_format = "Time is not in correct format- its must be in form of m (1m-59m), d (1d-30d), h (1-23h)"


def ML_DB(df_db):
    client=InfluxDBClient(host="3.135.200.69",port="8086")
    client.switch_database("mlmodel")

    for i in range(len(df_db)):
        time = df_db[df_db.columns[0]][i]
        field1value = df_db[df_db.columns[1]][i]
        json_body=[
        {
            "measurement": "MyResponseData7",
            "time": time,
            "fields": {
                "node_memory_Active_megabytes": field1value
                        }
        }]
        #print(json_body)
        client.write_points(json_body)

def ML_model_Timeseries(url_resp_out,predictionstep,predictioncount):
    zone = "Asia/Kolkata"
    pattern = '%Y-%m-%d %H:%M:%S'
    predictionstep = int(predictionstep)
    predictioncount = int(predictioncount)
    
    x =url_resp_out
    y = x['data']["result"][0]["values"]
    DataframeVar = x['data']['result'][0]['metric']['__name__']
    df_prep_old = pd.DataFrame(y)
    df_prep_old.columns=['Time',DataframeVar]
    df_prep_old = df_prep_old.apply(pd.to_numeric, errors='ignore')
    df_prep_old[DataframeVar] = df_prep_old[DataframeVar].multiply(1/1024000)
    df_prep_old['Time'] = df_prep_old['Time']+19800
    time_start = df_prep_old['Time'][0]
    time_end = df_prep_old['Time'][len(df_prep_old)-1]
    df_prep_old_list = []
    while (time_start <= time_end):
       df_prep_old_list.append(time_start)
       #1m=60
       time_start = time_start + 60
    df_prep_new = pd.DataFrame(df_prep_old_list,columns=['Time'])
    df = df_prep_new.merge(df_prep_old, how='left')
    df[DataframeVar].fillna((df[DataframeVar].mean()), inplace=True)
    df['Time']=pd.to_datetime(df['Time'],unit='s') #always convert time into date time format for ARIMA model
    df.set_index('Time',inplace=True)
    def adfuller_test(perf_act):
        result=adfuller(perf_act)
        labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
        for value,label in zip(result,labels):
            print(label+' : '+str(value) )
            if result[1] <= 0.05:
                print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
            else:
                print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
        
    adfuller_test(df[DataframeVar])
    secondDiff = 'Seasonal '+DataframeVar+' Difference'
    df[secondDiff]=df[DataframeVar]-df[DataframeVar].shift(1)
    adfuller_test(df[secondDiff].dropna())
    model=ARIMA(df[DataframeVar],order=(1,1,1))
    model_fit=model.fit()
    model=sm.tsa.statespace.SARIMAX(df[DataframeVar],order=(1, 1, 1),seasonal_order=(1,1,1,2))
    results=model.fit()
    start_data = int(len(df)*.85)
    end_data  = len(df)
    Log_Predict = DataframeVar+' Predict'
    Log_Predict_train = DataframeVar+' train'
    Log_Predict_test = DataframeVar+' test'
    Log_Predict_future = DataframeVar + ' predict future'
    df[Log_Predict]=results.predict(start=start_data,end=end_data,dynamic=True)
    Log_Predict_train = df[Log_Predict][start_data:end_data].values
    Log_Predict_test = df[DataframeVar][start_data:end_data].values
    # Accuracy metrics
    mape = np.mean(np.abs(Log_Predict_train - Log_Predict_test)/np.abs(Log_Predict_test))  # MAPE
    print("Accuracy " + str(100 - mape))
    print(df.index[0])
    print(df.index[-1])
    future_dates=[df.index[-1]+ DateOffset(minutes=x)for x in range(1,predictioncount+1,predictionstep)]
    #print(future_dates)
    #print('future_dates')
    future_datest_df=pd.DataFrame(index=future_dates,columns=df.columns)
    future_df=pd.concat([df,future_datest_df])
    future_df[Log_Predict_future] = results.predict(start = end_data, end = end_data+predictioncount, dynamic= True)
    Predict2json = future_df[Log_Predict_future][end_data:end_data+predictioncount].tolist()
    df_db = future_df[Log_Predict_future][end_data:end_data+predictioncount].to_frame()
    df_db.reset_index(inplace=True)
    df_db = df_db.rename(columns={'index': 'Time'})
    list_for_time = df_db[df_db.columns[0]].to_list()
    for i in range(len(list_for_time)):
        list_for_time[i]=int(time.mktime(time.strptime(str(list_for_time[i]), pattern)))
    list_for_value = df_db[df_db.columns[1]].to_list()
    #future_df.to_csv('file3.csv', header=True)
    #db call
    #print(Prediction2json)
    #print(type(Prediction2json))
    target = DataframeVar
    datapoints = dict(zip(list_for_time, list_for_value))
    print(len(Predict2json))
    print(len(future_dates))
    Prediction2json = {"target":target,"datapoints":datapoints}

    ML_DB(df_db)
        
    '''
    for row_index, row in df_db.iterrows():
        time = row[0]
        field1value = row[1]
        json_body=[
        {
            "measurement": "MyResponseData5",
            "time": time,
            "fields": {
                "node_memory_Active_megabytes": field1value
                        }
        }]
        print(json_body)
        client.write_points(json_body)
    '''
    return Prediction2json 

def Model_Selection(url_input_dict):
    ec2ip= "3.135.200.69:9090"
    print(ec2ip)
    #and url_input_dict['predict'] == 'timeseries'
    if 'predict' in url_input_dict.keys():
        parameter=url_input_dict['parameter']
        starttime=str(url_input_dict['starttime'])
        endtime=str(url_input_dict['endtime'])
        step=url_input_dict['step']
        predictionstep=url_input_dict['predictionstep']
        predictioncount=url_input_dict['predictioncount']
        #url = "http://"+ec2ip+"/api/v1/query_range?query="+parameter+"&start="+starttime+"&end="+endtime+"&step="+step+"&predictionstep="+predictionstep+"&predictioncount="+predictioncount
        url = "http://"+ec2ip+"/api/v1/query_range?query="+parameter+"&start="+starttime+"&end="+endtime+"&step="+step #"&predictionstep="+predictionstep+"&predictioncount="+predictioncount
        print("url for response")
        print(url)
        url="http://3.135.200.69:9090/api/v1/query_range?query=node_memory_Active_bytes&start=1598929170&end=1598965170&step=1m"
        return url,predictionstep,predictioncount

def URL_response(url_resp):
    try:
        url_resp = requests.get(url_resp)
        
        if url_resp.status_code != 200:
                # This means something went wrong.
            return "URL is not working"
            #raise ApiError('GET /tasks/ {}'.format(url_resp.status_code))
        else:
            print("Status code : 200, URL is working")

        url_resp_json = url_resp.json()

        print(url_resp_json)
        print(type(url_resp_json))
        print(len(url_resp_json["data"]["result"]))
    
        if len(url_resp_json["data"]["result"]) != 0:
            print("11111")
            return url_resp_json
        else:
            print("33333")
            return (error_message_time_format)
    except:
        print("2222")
        return (error_message_time_format)
    
  
def CalculateEpochFunc(CalculateEpoch):
    CalculateEpoch = CalculateEpoch
    CalculateEpoch = CalculateEpoch.lower()

    if len(CalculateEpoch)>0:
        
        CalculateEpoch_list = re.findall(r"[^\W\d_]+|\d+", CalculateEpoch)
        CalculateEpoch_list = CalculateEpoch_list[::-1]
        CalculateEpoch_dict = dict(itertools.zip_longest(*[iter(CalculateEpoch_list)] * 2, fillvalue=""))
        print(CalculateEpoch_dict)

        for i in range(len(CalculateEpoch_dict)):
            if len(CalculateEpoch_dict[list(CalculateEpoch_dict.keys())[i]]) == 0 :
                return(error_message_time_format)
            try: 
                CalculateEpoch_dict[list(CalculateEpoch_dict.keys())[i]] = int(CalculateEpoch_dict[list(CalculateEpoch_dict.keys())[i]])
            except:
                return (error_message_time_format)
    
        CalculateEpoch_check_in = list(CalculateEpoch_dict.keys())
        CalculateEpoch_check =['m', 'h', 'd']

        common_elements = set(CalculateEpoch_check).intersection(CalculateEpoch_check_in)
        #print(common_elements)

        if len(common_elements)>1 or len(common_elements) == 1 :
            common_elements_all = list(common_elements)
            for j in range(len(common_elements_all)):
                if list(common_elements)[j] == 'm':
                    if CalculateEpoch_dict[list(common_elements)[j]] > 0 and CalculateEpoch_dict[list(common_elements)[j]] <= 59:
                        CalculateEpoch_dict[list(common_elements)[j]] = CalculateEpoch_dict[list(common_elements)[j]]*60
                    else:
                        return (error_message_time_format)
                if list(common_elements)[j] == 'h':
                    if CalculateEpoch_dict[list(common_elements)[j]] > 0 and CalculateEpoch_dict[list(common_elements)[j]] <= 23:
                        CalculateEpoch_dict[list(common_elements)[j]] = CalculateEpoch_dict[list(common_elements)[j]]*3600
                    else:
                        return (error_message_time_format)
                if list(common_elements)[j] == 'd':
                    if CalculateEpoch_dict[list(common_elements)[j]] >0 and CalculateEpoch_dict[list(common_elements)[j]] <= 30:
                        CalculateEpoch_dict[list(common_elements)[j]] = CalculateEpoch_dict[list(common_elements)[j]]*86400
                    else:
                        return (error_message_time_format)
            
        else:
           return (error_message_time_format)

    #return(CalculateEpoch_dict)
    return(sum(CalculateEpoch_dict.values()))


def URL_extract(url_input_extract):
    print(url_input_extract)
    predict_all = ['predict', 'parameter', 'starttime', 'endtime', 'step', 'predictionstep', 'predictioncount']
    url_input_dict = dict(itertools.zip_longest(*[iter(re.split("&|=",url_input_extract))] * 2, fillvalue=""))
    check_parameters = all(item in predict_all for item in list(url_input_dict.keys()))
    print(url_input_dict)

    if check_parameters is True:
        output = URL_creation(url_input_dict)
        return output  
    else :
        return error_message_url_format

def URL_creation(url_input_dict):
    #print(url_input_dict)
    #print(CalculateEpochFunc(url_input_dict['starttime']))
    zone = "Asia/Kolkata"
    pattern = "%Y-%m-%d %H:%M:%S"
    url_input_dict['starttime'] = CalculateEpochFunc(url_input_dict['starttime'])
    
    if 'predictionstep' not in url_input_dict:
        url_input_dict['predictionstep'] = '1m'
    if 'predictioncount' not in url_input_dict:
        url_input_dict['predictioncount'] = '20'
    if 'endtime' not in url_input_dict:
        end_time = int(time.mktime(time.strptime(datetime.now(timezone(zone)).strftime(pattern), pattern)))
        url_input_dict['endtime'] = end_time
    else:
        try:
            url_input_dict['endtime']= CalculateEpochFunc(url_input_dict['endtime'])
        except:
            return (error_message_time_format)
        end_time = int(time.mktime(time.strptime(datetime.now(timezone(zone)).strftime(pattern), pattern)))
        url_input_dict['endtime'] = end_time - url_input_dict['endtime']

    url_input_dict['starttime'] = url_input_dict['endtime'] - url_input_dict['starttime']###

    if url_input_dict['starttime'] > url_input_dict['endtime'] or url_input_dict['starttime'] == url_input_dict['endtime']:
        return "url starttime and endtime cant be same"
    
    print(url_input_dict)

    url_out,predictionstep,predictioncount = Model_Selection(url_input_dict)
    print(url_out)
    print(predictionstep)
    print(predictioncount)
    url_resp_out = URL_response(url_out)
    print(url_resp_out)
    print("time series")
   
    if type(url_resp_out) == dict:
        Ml_mode_out_for_db = ML_model_Timeseries(url_resp_out,predictionstep,predictioncount)
        return Ml_mode_out_for_db
    else:
        return "data is not available"
   
         

app = Flask(__name__)

@app.route('/')
def home_ML():
    return '200 OK!! ML server is UP'

@app.route('/db')
def home_DB():

    return 'Connected to DB'

@app.route('/url/<name>')
def new_func(name):
    try:
        url_input = URL_extract(name)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type)
        print(fname)
        print(exc_tb.tb_lineno)
        return error_message_time_format  
    return url_input

#http://127.0.0.1:5000/query/?url=https://google.com
@app.route('/query/')
def func1():
    query = request.args.get('url')
    return ML_code(query)

if __name__ == '__main__':
    app.run(debug=True)
