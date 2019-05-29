import os
import pandas as pd
import pickle
import traceback
from bs4 import BeautifulSoup   
from datetime import datetime


def pickle_parser(directory):
    for item in directory:
        current_pickle = pickle.load(open(f'D:\pickle_jar\{item}','rb')) 
        urban_dict = {}
        start_time = datetime.now()
        
        for key, value in current_pickle.items():
            
            try:
                soup = BeautifulSoup(value, 'lxml') 
                s = soup.find_all("div", class_="def-panel")[0]
                top_def = s.find("div", class_="meaning").text
                author = s.find("div", class_="contributor").find("a").text
                post_date = datetime.strptime(s.find("div", class_="contributor").find("a").next_sibling.strip(), '%B %d, %Y')
                upvotes = int(s.find("a", class_="up").text)
                downvotes = int(s.find("a", class_="down").text)
                examples = str(s.find("div", class_="example").text).strip('"')
                try:
                    tags = s.find("div", class_="tags").text
                except:
                    tags = None
                urban_dict[key] = {'top_def': top_def, 'author': author, 
                          'post_date': post_date, 'upvotes': upvotes, 
                          'downvotes': downvotes, 'tags': tags, 'example': examples}
                
            except Exception:
                print(key)
                traceback.print_exc()
                pass
            
            except KeyboardInterrupt:
                print('You broke it!')
                break
                
        # pickle out parsed chunk
        pickle_out = open(f'parsed_{item}','wb')
        pickle.dump(urban_dict, pickle_out)
        pickle_out.close()
        
        end_time = datetime.now()
        parse_time = (end_time - start_time)
        print(f'Parsing chunk {item} took {parse_time}')

def assemble_grand_dict(directory):
    grand_dict = {}
    for item in directory:
        parsed_pickle = pickle.load(open(f'{item}','rb')) 
        for key, value in parsed_pickle.items():
            grand_dict[key] = value
            
    pickle_out = open(f'grand_dict','wb')
    pickle.dump(grand_dict, pickle_out)
    pickle_out.close()
        
    
    return grand_dict

def dict_to_df(input_dict):
    return pd.DataFrame.from_dict(input_dict, orient='index')
        
if __name__ == '__main__':
    pickle_parser(os.listdir('D:\pickle_jar'))
    grand_dict = assemble_grand_dict(os.listdir(os.getcwd()))
    master = dict_to_df(grand_dict)
    master.to_csv('master_data.csv')
