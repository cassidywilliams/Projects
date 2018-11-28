import pandas as pd
import numpy as np
import fxcmpy
from datetime import datetime, timedelta
from __future__ import print_function
from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools

# http://www.histdata.com/f-a-q/
#define number of lags to test for features. if this is the method determined to work best, add a loop to automate select number of lags
lags = 10
symbol = 'EUR/USD'

api = fxcmpy.fxcmpy(config_file='fxcm.cfg')
start = datetime.strftime(datetime.now() - timedelta(15), '%Y-%m-%d')
end = datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d')

#get l14d minute data for symbol
try:
    raw = api.get_candles(symbol, period='m1', start=start, end=end)
    raw.info()
except ValueError:
    print("Cannot connect to API")

raw.reset_index(inplace=True)
data = pd.DataFrame()
data['date'] = [d.date() for d in raw['date']]
data['time'] = [d.time() for d in raw['date']]
data['close'] = (raw['bidclose'] + raw['askclose']) / 2

piv = data.pivot(index=str('time'), columns=str('date'), values='close')
#drop days with missing data > threshold
piv.dropna(inplace=True, axis=1, thresh=len(piv)*.85)
#piv.reset_index(level=0, inplace=True)

piv.plot(legend=None)

# pass in a pivot table with minute/hour data for index and date for column headers
     
def optimal_finder(piv):
    global results_3, tester
    open_list = []
    close_list = []
    delta_list = []          
                
    for column in piv:
        for index, row in piv.iterrows():
            x = row[column]
            x_index = index
            for index, row in piv.iterrows():
                y = row[column] 
                delta = y/x
                y_index = index
                open_list.append(x_index)
                close_list.append(y_index)
                delta_list.append(delta)
    
    results = pd.DataFrame(
            {'open' : open_list,
             'close' : close_list,
             'difference' : delta_list})
        
    results_no_null = results.fillna(0)
    
    results_3 = results_no_null.groupby(['open', 'close'], as_index=False)['difference'].mean()
    
    tester = results_3.loc[results_3['close'] > results_3['open']]
    print(tester.loc[tester['difference'][::-1].idxmax()])    
       
optimal_finder(piv)
opt = str(tester.loc[tester['difference'][::-1].idxmax()]) 

import httplib2
import os

from apiclient import discovery
import oauth2client
from oauth2client import client
from oauth2client import tools

try:
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

SCOPES = 'https://mail.google.com/'
CLIENT_SECRET_FILE = 'credentials.json'
APPLICATION_NAME = 'Gmail API Quickstart'


def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'gmail-quickstart.json')

    store = oauth2client.file.Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else: # Needed only for compatability with Python 2.6
            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials

import base64
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import mimetypes
from httplib2 import Http

from apiclient import errors

from apiclient.discovery import build
credentials = get_credentials()
service = build('gmail', 'v1', http=credentials.authorize(Http()))

def SendMessage(service, user_id, message):
  """Send an email message.

  Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    message: Message to be sent.

  Returns:
    Sent Message.
  """
  try:
    message = (service.users().messages().send(userId=user_id, body=message)
               .execute())
    print('Message Id: %s' % message['id'])
    return message
  except:
    print('An error occurred: %s')


def CreateMessage(sender, to, subject, message_text):
  """Create a message for an email.

  Args:
    sender: Email address of the sender.
    to: Email address of the receiver.
    subject: The subject of the email message.
    message_text: The text of the email message.

  Returns:
    An object containing a base64 encoded email object.
  """
  message = MIMEText(message_text)
  message['to'] = to
  message['from'] = sender
  message['subject'] = subject
  return {'raw': base64.urlsafe_b64encode(message.as_string().encode()).decode()}


message = CreateMessage('casswllms@gmail.com', 'benrynlds@gmail.com', 'Your daily predictions are here!', opt)
SendMessage(service, 'me', message)