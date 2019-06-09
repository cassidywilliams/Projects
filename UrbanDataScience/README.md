# UrbanDataScience

The purpose of this project was analyze the words on urbandictionary.com to understand the relationships between words, as well as what leads some words to be more popular than others. This required the development of an asynchronous web scraper to collect the data, which was then later analyzed and modeled.

The purpose of this project was to collect, analyze, and model the 1.9 million words found on [Urban Dictionary](https://www.urbandictionary.com/). This project contains several distinct components:

* **Data acquisition:** The data was acquired by use of an asynchronous web scraper that I built to collect the HTML for all 1.9 million words. The data is pickled into chunks of 15,000 words, as a measure to avoid memory errors, as this was ran on a local machine. The asynchronous scraper reduced data collection time from over 130 hours to under three.

* **Data parsing and cleaning:** Once the HTML data was stored in a collection of pickle files, each one was parsed to strip out only the information necessary for this project, which includes: word, definition , author, post date, and hashtags. The parsed data is then combined into one .csv to easily open and manipulate. Rows with strange values and nulls are removed.

* **Feature engineering:** In order to begin modeling the data, I created several features that could be useful in understanding relationships between variables and to ultimate answer the question of what makes one word more popular than another. Some features that were explored:
  * Length of definition
  * Length of example
  * Posts by author
  * Posts by date
  * Number of days online
  * Probability of word containing profanity
  * Probability of definition containing profanity
  * Probability of example containing profanity
  * Combined popularity score of all hashtags used
  * Total interactions (upvotes + downvotes)
  * Interaction polarity (how negative or positive it was voted)
  * Days featured (if it was featured on home page)
  
* **Exploratory data analysis:** To begin to understand the distributions of each input variable and their relationships, I spent some time analyzing this data. More to come on this later.

## Authors

* **Cassidy Williams** - [cassidywilliams](https://github.com/cassidywilliams)

