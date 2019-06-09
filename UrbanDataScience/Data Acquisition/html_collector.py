import asyncio
import aiohttp
from datetime import datetime
import traceback
from itertools import islice
import pickle

# Open pickle file containing all words and links.
try:
    words_and_links
except NameError:  
    words_and_links = pickle.load(open('words_and_links.pickle','rb')) 


# Multi-threading functions
async def get_html(session, word, url):
    async with session.get(f'https://www.urbandictionary.com{url}', ssl=False) as response:
        return word, await response.text()


async def fetch_all(url_dict, loop):
    async with aiohttp.ClientSession(loop=loop) as session:
        results = await asyncio.gather(*[get_html(session, *item) for item in url_dict.items()], return_exceptions=True)
        return dict(results)


def chunks(data, chunk_size=10000):
    
    '''Break large dict of words and links into list of dicts for easier consumption.'''
    
    it = iter(data)
    for i in range(0, len(data), chunk_size):
        yield {k:data[k] for k in islice(it, chunk_size)}


if __name__ == '__main__':
    
    list_of_dicts = list(chunks(words_and_links, 15000))    
    
    for chunk in list_of_dicts:
        start_time = datetime.now()
        # get async requests and create dict of the htmls
        try:
            loop = asyncio.get_event_loop()
            htmls = loop.run_until_complete(fetch_all(chunk, loop))
            print(f'successfully received htmls for chunk: {list_of_dicts.index(chunk)}')
        except Exception:
            traceback.print_exc()
            pass
        
        except KeyboardInterrupt:
            print('You broke it!')
        
        # pickle out chunks of htmls
        pickle_out = open(f'D:/pickle_jar/html_{list_of_dicts.index(chunk)}.pickle','wb')
        pickle.dump(htmls, pickle_out)
        pickle_out.close()
        print(f'successfully pickled chunk: {list_of_dicts.index(chunk)}')        
        end_time = datetime.now()
        time_per_element = (end_time - start_time)/len(htmls)
        print('time per element: ', time_per_element)
        del htmls

