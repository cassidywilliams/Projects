import time
import datetime
import pandas as pd
from selenium import webdriver
import configparser
import logging 
#from fbprophet import Prophet
import numpy as np


chrome_driver_executable = 'C:\\Users\\Cass\\AppData\\Local\\Programs\\Python\\Python36\\Scripts\\chromedriver.exe'
driver = webdriver.Chrome(chrome_driver_executable)
driver.get("https://www.etoro.com/markets/eurusd")

# PARSE CONFIG FILE HERE AND LOGIN TO ACCOUNT
def login():
    config = configparser.ConfigParser()
    config.read("C:\Windows\System32\drivers\etc\config.txt")
    pwd = config.get("etoro", "password")
    user = config.get("etoro","email")
    driver.get("https://www.etoro.com/login")
    driver.find_element_by_id('username').send_keys(user)
    driver.find_element_by_id('password').send_keys(pwd)
    
login()
# WRAP THIS ALL IN A GET_DATA FUNCTION
dates = []
bid = []
ask = []

for i in range(0,100):
    while True:
        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_val = driver.find_element_by_xpath("//*[contains(@class, 'head-info-stats-value ng-binding')]")
            time.sleep(3)
            dates.append(ts)
            ask.append(float(current_val.text))
            bid.append(float(current_val.text) + .0003)
            print(f'{ts} EURUSD ask: {current_val.text} bid: {float(current_val.text) + .0003}')
        except:
            continue
        break
driver.close()

rates = pd.DataFrame(
    {'ts': dates,
     'ask': ask,
     'bid': bid
     })
    
rates2 = pd.DataFrame(
    {'ds': dates,
     'ask': ask,
     'bid': bid
     })
rates2['y'] = np.log(rates2['ask'])   
 
    
# ANALYSIS PORTION: TIME SERIES WITH PROPHET?   
m = Prophet()
m.fit(rates2)    
future = m.make_future_dataframe(periods=10, freq='S')
forecast = m.predict(future)  
fig1 = m.plot(forecast)


# ADD DECISION MAKER

def execute_trade():  
    trade_button = driver.find_element_by_xpath("//*[contains(@class, 'ng-scope button-standard button-blue head-action-button')]")
    trade_button.click()

#CHOOSE TRADE TYPE: BUY OR SELL
#INPUT AMOUNT TO TRADE
#CLICK ON AND ADJUST TAKE PROFIT
#CLICK ON AND ADJUST STOP LOSS
#SCRAPE OVERNIGHT FEE (AND MAYBE WEEKEND FEE)
    
    open_trade_button = driver.find_element_by_xpath("//*[contains(@class, 'execution-button button-standard button-blue pointer ng-scope')]")
    open_trade_button.click()


# CLOSING TRADES