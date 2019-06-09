import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
   

def get_words_and_dates(last_page):
    words = []
    dates = []
    pages = []
    for page in range(1, last_page +1):
        try:
            response = requests.get(f'https://www.urbandictionary.com/?page={page}')  
            soup = BeautifulSoup(response.text, 'lxml')
            for tag in soup.find_all("div", class_="def-panel"):
                words.append(tag.find("a", class_="word").text)
                dates.append(tag.find("div", class_="ribbon").text)
                pages.append(page)
        except:
            pass
        
    df = pd.DataFrame({'word': words, 'date': dates, 'page': pages})
    
    return df

def date_convert(str_month_day, int_year):
    date = str_month_day + " " + str(int_year)
    
    return datetime.strptime(date.strip(), '%b %d %Y')


def date_creator(df):
    df['date_shifted'] = df['date'].shift()  
    df['est_year'] = None
    df['est_year'][0] = 2019    
    for i in range(1, len(df)):
        if df.loc[i, 'date'][:3] == 'Dec' and df.loc[i, 'date_shifted'][:3] == 'Jan':
           df.loc[i, 'est_year'] = df.loc[i-1, 'est_year'] -1
        else:
          df.loc[i, 'est_year'] = df.loc[i-1, 'est_year']    
    df['date_formatted'] = df.apply(lambda x: date_convert(x['date'], x['est_year']), axis=1)
    df['days_featured'] = df['date_formatted'].apply(lambda x: (scrape_date - x).days)
    
    return df


if __name__ == '__main__':
    scrape_date = datetime.strptime('2019-05-01', '%Y-%m-%d')
    word_of_the_day = get_words_and_dates(711)
    date_creator(word_of_the_day)
    word_of_the_day = word_of_the_day[word_of_the_day.days_featured >= 0]
    # currently only consider first date of feature. a small subset of words have two features, which may be addressed later
    word_of_the_day.drop_duplicates(subset=['word'], keep='last', inplace=True)
    word_of_the_day.to_csv('word_of_the_day.csv', index=False)

    


