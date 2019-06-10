import string
import pandas as pd
from collections import Counter
from datetime import datetime
from profanity_check import predict_prob
from spellchecker import SpellChecker




def tag_parser(string_of_tags):
    
    'This parses a string of hashtags into a list.'

    try:
        x = string_of_tags.split('#')
        clean_tag_list = []
        for tag in x:
            if tag != '' and tag != ' ' and tag is not None:
                clean_tag_list.append(tag.strip())
        return clean_tag_list
    except:
        return string_of_tags                       


def spell_checker(input_string):
    
    """This checks if the input word or string is contained in an English 
        dictionary. If a string is passed as an input, each word is checked.
        Returns a percentage of English words."""
    
    spell = SpellChecker()
    try:
        input_string = [char for char in input_string if char in string.ascii_letters + ' ' + '-' + "'"]
        input_string = ''.join(input_string)
        input_string = input_string.replace("-", " ")
    except:
        return None
    i = 0
    input_len = len(input_string.split())
    if input_len > 0:
        for word in input_string.split():
            try:
                if len(word) == 1:
                    if word.lower() == 'a' or word.lower() == 'i':
                        i += 1
                elif word in spell:
                    i += 1
                else:
                    pass
            except:
                pass
        return i/input_len
 
    
def len_except(code):
    
    'Simply checks length of input, but includes exception handling.'
    
    if isinstance(code, list):
        return len(code)
    else:
        try:  
            l = len(code.split())
            return l
        except:
            pass
    
    
def posts_by_auth(df):
    
    'Calculates number of posts by author. Returns dictionary.'
    
    return dict(df['author'].value_counts())


def posts_by_date(df): 
    
    'Calculates number of posts by date. Returns dictionary.'
    
    return dict(df['post_date'].value_counts())


def tag_usage_counter(df):
    
    'Used to count occurence of each hashtag.'
    
    master_tags = []
    for index, row in df.iterrows():
        try:
            for i in row['tag_list']:
                master_tags.append(i)
        except:
            pass
    tag_counts = Counter(master_tags)
        
    return tag_counts  


def tag_prep(df):
    
    'Adds two columns to the main dataframe: a list of tags and count of tags.'
    
    df['tag_list'] = df['tags'].apply(lambda x:tag_parser(x))
    df['tag_count'] = df['tag_list'].apply(lambda x: len_except(x))
    
    return df

def create_tag_db(df):
    
    """Creates dataframe containing each unique hashtag, number of uses, and
        tag usage rate."""
    
    tag_dict = tag_usage_counter(df)
    tag_data = pd.DataFrame.from_dict(tag_dict, orient='index', columns=['uses'])
    tag_data['tag_usage_rate'] = tag_data['uses'].apply(lambda x: x/len(df))
    
    return tag_data  


def tag_usage_rate_calc(string_of_tags, tag_df):
    
    'Creates total tag usage rate value for a list of tags.'
   
    tag_score = 0
    try:
        for i in string_of_tags:
            tag_score += tag_df.loc[i]['tag_usage_rate']  
    except:
        pass
    
    return tag_score


def profanity(string):
    
    'Returns probability of word or string containing profanity.'
    
    try:
        return predict_prob([string])[0]
    except:
        None


def binary_contains(test, string):
    
    'Used to test if a list contains a string, returns binary.'
    
    if string != string:
        return 0
    else:
        if test in string:
            return 1
        else:
            return 0


def feature_fab(df, auth_dict, date_dict, tag_data, words_of_the_day, scrape_date):
    
    """This function creates many of the features columns."""
    
    df['def_length'] = df['top_def'].apply(lambda x: len_except(x))
    df['example_length'] = df['example'].apply(lambda x: len_except(x))  
    df['posts_by_auth'] = df['author'].apply(lambda x: auth_dict[x])
    df['posts_by_date'] = df['post_date'].apply(lambda x: date_dict[x])
    df['days_online'] = df['post_date'].apply(lambda x: (scrape_date - datetime.strptime(x, '%Y-%m-%d')).days)
    df['word_profanity_prob'] = df['word'].apply(lambda x: profanity(x))
    df['def_profanity_prob'] = df['top_def'].apply(lambda x: profanity(x))
    df['example_profanity_prob'] = df['example'].apply(lambda x: profanity(x))
    df['total_tag_usage_rate'] = df['tag_list'].apply(lambda x: tag_usage_rate_calc(x, tag_data))
    df['total_interactions'] = df['upvotes'] + df['downvotes']
    df['interaction_polarity'] = (df['upvotes'] - df['downvotes'])/df['total_interactions']
    df['days_featured'] = df['word'].map(dict(zip(words_of_the_day.word, words_of_the_day.days_featured)))
    
    return df

def top_n_tag_tester(top_n_tags, df):
    
    """ This function creates a column for each tag in the top_n_tags df,
        which are then used to test if the word was tagged with it (binary)."""

    for i, row in top_n_tags.iterrows():
        df[f'is_{i}'] = None       
    
    for col in df.columns:
        if col[:3] == 'is_':
            df[col] = df['tag_list'].apply(lambda x: binary_contains(col[3:], x))        

    return df

if __name__ == "__main__":
    scrape_date = datetime.strptime('2019-05-01', '%Y-%m-%d')
    data = pd.read_csv('master_data.csv')
    words_of_the_day = pd.read_csv('word_of_the_day.csv')
    data.rename(columns = {'Unnamed: 0':'word'}, inplace=True)
    data.dropna(subset=['author'], inplace=True)
    data = tag_prep(data)
    auth_dict = posts_by_auth(data)
    date_dict = posts_by_date(data)
    tag_data = create_tag_db(data)
    top_n_tags = tag_data.nlargest(100, 'tag_usage_rate')
    data = feature_fab(data, auth_dict, date_dict, tag_data, words_of_the_day, scrape_date)
    data = top_n_tag_tester(top_n_tags, data)
    data.to_csv('engineered_data.csv', index=False)
