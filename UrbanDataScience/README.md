# UrbanDataScience

The purpose of this project was to collect, analyze, and model the 1.9 million words found on [Urban Dictionary](https://www.urbandictionary.com/). This project contains several distinct components:

* **Data Acquisition:** The data was acquired by use of an asynchronous web scraper that I built to collect the HTML for all 1.9 million words. The data is pickled into chunks of 15,000 words, as a measure to avoid memory errors, as this was ran on a local machine. The asynchronous scraper reduced data collection time from over 130 hours to under three.

* **Data Parsing and Cleaning:** Once the HTML data was stored in a collection of pickle files, each one was parsed to strip out only the information necessary for this project, which includes: word, definition, author, post date, hashtags and other metadata. The parsed data is then combined into one .csv to easily open and manipulate. Rows with strange values and nulls are removed.

* **Feature Engineering:** In order to begin modeling the data, I created several features that could be useful in understanding relationships between variables and to ultimately answer the question of what makes one word more popular than another. Some features that were explored:
  * Length of definition
  * Length of example
  * Posts by author
  * Posts by date
  * Number of days online
  * Probability of word containing profanity
  * Probability of definition containing profanity
  * Probability of example containing profanity
  * NLTK columns: nouns, verbs, adjective counts for word, definition, and example
  * Popularity score of hashtags used
  * Total interactions (upvotes + downvotes)
  * Interaction polarity (how negative or positive it was voted)
  * Days featured (if it was featured on home page)
  * Encoded variables (each column indicating one of top 100 hashtags used)
  
* **Exploratory data analysis:** To begin to understand the distributions of each input variable and their relationships, I spent some time analyzing this data. More to come on this later.

## Authors

* **Cassidy Williams** - [cassidywilliams](https://github.com/cassidywilliams)

