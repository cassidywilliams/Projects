import urllib.request
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import os
from os import path
from wordcloud import WordCloud
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from tika import parser

raw = parser.from_file('LPTHW.pdf')
text = raw['content']
sentences = sent_tokenize(text)
words = word_tokenize(text)

#read html of website, stip html tags, and convert text to tokens
#response = urllib.request.urlopen('https://www.bbc.com/news')
#html = response.read() 
#soup = BeautifulSoup(html,"html5lib") 
#text = soup.get_text(strip=True) 
tokens = [t for t in text.split()]
#tokens = nltk.word_tokenize(text)
tokens=[token.lower() for token in tokens if token.isalpha()]



#duplicate tokens and remove stop words (of, the, a, etc)
clean_tokens = tokens[:] 
sr = stopwords.words('english') 
for token in tokens: 
    if token.lower() in stopwords.words('english'): 
        clean_tokens.remove(token)   
        
cloud_txt = " ".join(clean_tokens)
    
#determine frequency of tokens now that stop words are excluded    
freq = nltk.FreqDist(clean_tokens)

#print frequency of each word, then plot on frequency plot 
for key,val in freq.items(): 
    print(str(key) + ':' + str(val))
freq.plot(20,cumulative=False)

wordcloud = WordCloud().generate(cloud_txt)
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


#wordcloud = WordCloud(max_font_size=40).generate(cloud_txt)
#plt.figure()
#plt.imshow(wordcloud, interpolation="bilinear")
#plt.axis("off")
#plt.show()




