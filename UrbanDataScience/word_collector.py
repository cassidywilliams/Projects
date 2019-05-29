import requests
import pickle
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def get_page_counts():
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    browser = webdriver.Chrome(chrome_options=chrome_options)
    
    last_pages_dict = dict.fromkeys([chr(x) for x in range(ord('A'), ord('Z') + 1)])
    last_pages_dict['*'] = None
    for letter in last_pages_dict:
        browser.get(f'https://www.urbandictionary.com/browse.php?character={letter}&page=2')
        last_pages_dict[letter] = int(browser.find_element_by_xpath("//*[@id='content']//*[contains(text(), 'Last')]").get_attribute('href').split('page=')[1])
    
    return last_pages_dict
    

def get_words_and_links(last_pages):
    words_and_links = {}
    for letter in last_pages:
        for page in range(1, last_pages[letter]+1):
            try:
                response = requests.get(f'https://www.urbandictionary.com/browse.php?character={letter}&page={page}')  
                print(f'gathered words for {letter}, page {page}')
                soup = BeautifulSoup(response.text, 'lxml')
                for tag in soup.find('ul', {'class': 'no-bullet'}).find_all('li'):
                    words_and_links[tag.text] = tag.a['href']
            except:
                pass
    words_and_links = {k.strip():v for k, v in words_and_links.items()}   
     
    return words_and_links


if __name__ == '__main__':
    last_pages = get_page_counts()
    words_and_links = get_words_and_links(last_pages)
#    pickle_out = open(f'last_pages.pickle','wb')
#    pickle.dump(last_pages, pickle_out)
#    pickle_out.close()
    pickle_out = open(f'words_and_links.pickle','wb')
    pickle.dump(words_and_links, pickle_out)
    pickle_out.close()
    
