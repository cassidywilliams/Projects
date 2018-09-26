import time
import datetime
import pandas as pd
from selenium import webdriver
import configparser
import logging 
#from fbprophet import Prophet
import numpy as np


chrome_driver_executable = 'C:\\Users\\Cass\\AppData\\Local\\Programs\\Python\\Python36\\Scripts\\chromedriver.exe'
options = webdriver.ChromeOptions() 
options.add_argument("user-data-dir=C:\\Users\\Cass\\AppData\\Local\\Google\\Chrome\\User Data")
options.add_argument("--no-sandbox") 
driver = webdriver.Chrome(executable_path=chrome_driver_executable, options=options)

def login():
    """
    This function parses the config file for login information then logs in to the etoro account.
    """
    config = configparser.ConfigParser()
    config.read("C:\Windows\System32\drivers\etc\config.txt")
    pwd = config.get("etoro", "password")
    user = config.get("etoro","email")    
    driver.get("https://www.etoro.com/login")
    time.sleep(2)
    
    while True:   
        try:
            driver.find_element_by_xpath("//*[contains(@class, 'i-menu-user-username')]")
        except:
            login_box = driver.find_element_by_id('username')
            login_box.clear()
            login_box.send_keys(user)
            driver.find_element_by_id('password').send_keys(pwd)
            time.sleep(2)
            driver.find_element_by_xpath("//*[contains(@class, 'big wide dark pointer')]").click()
            time.sleep(2)
            try:
                driver.find_element_by_xpath("//*[contains(@class, 'popover-close-button')]").click()
            except:
                pass
            time.sleep(3)
            continue
        else:
            print('Successfully logged in!')
            break

def logout():
    try:
        driver.find_element_by_xpath("//*[contains(@class, 'i-menu-icon sprite logout')]").click()
        print('Successfully logged out.')
    except:
        print('You are not logged in.')

def getdata(pair, frequency):
    driver.get("https://www.etoro.com/markets/" + pair)
    dates = []
    bid = []
    ask = []
    
    for i in range(0,2):
        while True:
            try:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                current_val = driver.find_element_by_xpath("//*[contains(@class, 'head-info-stats-value ng-binding')]")
                time.sleep(frequency)
                dates.append(ts)
                ask.append(float(current_val.text))
                bid.append(float(current_val.text) + .0003)
                print(f'{ts} EURUSD ask: {current_val.text} bid: {float(current_val.text) + .0003}')
            except:
                continue
            break

    rates = pd.DataFrame(
        {'ds': dates,
         'ask': ask,
         'bid': bid
         })
    rates['y'] = np.log(rates['ask'])   
    return rates

def check_equity():
    global current_equity
    current_equity = driver.find_element_by_xpath("//div[@class='w-footer-unit total active']/span[@class='w-footer-unit-value ng-binding']").text
    return current_equity

def check_available():
    global funds_available
    funds_available = driver.find_element_by_xpath("//div[@class='w-footer-unit balance']/span[@class='w-footer-unit-value ng-binding']").text
    return funds_available

def check_allocated():
    global allocated
    allocated = float(driver.find_element_by_xpath("//div[@class='w-footer-unit amount']/span[@class='w-footer-unit-value ng-binding']").text[1:])
    return allocated

def check_profit():
    global profit
    profit = driver.find_element_by_xpath("//div[@class='w-footer-unit profit']/span[contains(@class, 'w-footer-unit-value ng-binding')]").text
    return profit

def edit_position(pair, take_profit):
    driver.get("https://www.etoro.com/portfolio/" + pair)
    time.sleep(3)
    #adjust take profit
    driver.find_element_by_xpath("//div[@class='e-btn light i-ptc-action edit ng-scope']").click()
    driver.find_element_by_xpath("//div[@class='box-tab-value ng-binding ng-scope positive']").click()
    time.sleep(2)
    driver.find_element_by_xpath("//div[@class='stepper-switch']").click()
    value = driver.find_element_by_xpath("//*[contains(@class, 'stepper-value')]")
    value.clear()
    value.send_keys(str(allocated*float(take_profit)))
    time.sleep(3)   
    driver.find_element_by_xpath("//button[contains(@class, 'execution-button')]").click()
    time.sleep(1)   
    driver.find_element_by_xpath("//button[contains(@class, 'execution-button')]").click()
    print("Position successfully updated")
    
def open_trade(pair, stop_loss, take_profit, trade_type, amount):
    driver.get("https://www.etoro.com/markets/" + pair)
    time.sleep(2)
    while True:
        try:
            driver.find_element_by_xpath("//*[contains(@class, 'stepper-value')]")
        except:
            driver.find_element_by_xpath("//div[@class='ng-scope button-standard button-blue head-action-button']").click()
            continue
        else:
            break        
    value = driver.find_element_by_xpath("//*[contains(@class, 'stepper-value')]")
    value.clear()
    value.send_keys(amount)
    
//*[@id="open-position-view"]/div[2]/div/div[2]/div[2]/div[1]/div[2]/input
//*[@id="open-position-view"]/div[2]/div/div[3]/tabs/div[3]/tabscontent/tab[3]/div/div/div/div[1]/div[2]/input
    


    
    
login()    
check_equity()
check_available()
check_allocated()
check_profit()    
edit_position("eurusd", 0.1)    
open_trade("eurusd",1,1,1,200)







driver.close()
 
    
# ANALYSIS PORTION: TIME SERIES WITH PROPHET?   
m = Prophet()
m.fit(rates)    
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