
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/how-small-business-can-survive-the-coronavirus-crisis/'

# Make a request to the URL
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Extract the article title
title = soup.find('h1').text

# Extract the article text
article_text = ''
for paragraph in soup.find_all('p'):
    article_text += paragraph.text + '\n'

# Print the title and article text
print(title)
print(article_text)

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# example text
text = ('''How Small Business can survive the Coronavirus Crisis
The word pandemic may be the most used word for the last decade, and we know why. The year 2020 comes with a lot of unprecedented threats to humans as well as to nature. While no business was left insulated from its effect, small businesses were one of the most affected, as they rely on day to day sales and keep the inventory minimum, based on the demand of customers and have a low margin to maneuver.
In an article by HSBC’s Navigator, which described the preparedness and contingency plan for the future, Indian business rated high among its peers. The businesses in India have shown resilience to face adversity which can be attributed to varying degrees of adjustment and adaptability. The businesses, while able to adhere to government guidelines have managed to facilitate their customers efficiently and effectively. However, the impact on micro, small, and medium enterprises cannot be left unattended. 
The Indian subcontinent has an estimated ₹ 633.88 lakh of MSME, out of which 51.2% are in rural areas which employ 44.84% of the total employment provided by this Sector. The MSME sector with 1170 lakh people constitutes 40% of the total workforce. As per the ministry of commerce MSME contribution in Gross Domestic Product (GDP) and export are 37% and 43% respectively.
With the huge contribution toward GDP as well as export, the impact on the national economy due to the shock in the MSME sector from the pandemic will be equally devastating. While different businesses face different types of obstacles, we have pointed out 8 of the most common issues faced by them.
For any business, cash in hand is oxygen. The small business which usually has at most 1-month cash to run the business if the revenue stream dries up, which in this tough time has to improve gradually. To manage the optimal cash balance is the key to success in this tough time. The optimal cash balance here, defined as “the less of the amount of cash outflow in form rent, employee salaries, and raw material without postponing the payable as they are doing what banks have to do, to the inflow from sale”. 
In any given situation, business is like juggling between multiple tasks to optimize the performance while maximizing the revenue. The current situation has added another layer of challenge. This challenge will test management skills in various aspects of our business owner. 
With this article, we have pointed out the major challenges and possible ways to alleviate the impact. As Anthony Robbins quotes, “Every problem is a gift—without problems we would not grow”, the current market has wiped out their share in the market which gives an opportunity to other businesses to grow and lower the expense.
While there may be many facets that need to be handled precisely, communication is one of the major tools that will create a positive environment among workers or teams which will influence the customer behavior and result in terms of revenue. The communication will also help to increase productivity and create trust in a supply chain which will improve the cash conversion cycles and working capital ratio.
Marketing through word of mouth has no added cost but the benefit can be scaled too much-required revenue. The choice of an effective marketing tool can also influence our target customer and will influence their will to buy the product or service(s) strategically.
Cash from operating activity improves if we were able to lower our expenses. To lower these expenses, small business owners have to create a different plan (good, average, and worst) keeping the 6-month frame in mind. Breaking down them in monthly targets would further help them to identify the actual scenario and update the strategy accordingly.  
As Warren Buffet quotes, “You only have to do a few things right in your life so long as you don’t do too many things wrong.” this is the right time to do the right thing and over time businesses will ripe its benefits. 
''')
scores = sid.polarity_scores(text)

# extract the positive and negative scores
positive_score = scores['pos']
negative_score = scores['neg']

print("Positive Score:", positive_score)
print("Negative Score:", negative_score)


# calculate the sentiment scores for each sentence
scores = sid.polarity_scores(text)

# extract the compound polarity score
polarity_score = scores['compound']

print("Polarity Score:", polarity_score)

from textblob import TextBlob

blob = TextBlob(text)
subjectivity_score = blob.sentiment.subjectivity

print("Subjectivity Score:", subjectivity_score)

import nltk
from nltk.tokenize import sent_tokenize

# example text

# tokenize the text into sentences
sentences = sent_tokenize(text)

# calculate the average sentence length
avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences)

print("Average Sentence Length:", avg_sentence_length)

import nltk
from nltk.corpus import cmudict


# Load the dictionary
d = cmudict.dict()

def count_syllables(word):
    """
    Count the number of syllables in a word.
    """
    try:
        return max([len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]])
    except KeyError:
        # if word not found in the dictionary, assume it has one syllable
        return 1

def is_complex(word):
    """
    Determine whether a word is complex based on the number of syllables.
    """
    return count_syllables(word) >= 3

def percentage_complex_words(text):
    """
    Calculate the percentage of complex words in a text.
    """
    words = nltk.word_tokenize(text.lower())
    num_words = len(words)
    num_complex_words = len([word for word in words if is_complex(word)])
    return 100 * num_complex_words / num_words


print('percentage_complex_words',percentage_complex_words(text)) # Output: 60.0

from textstat import textstat
fog_index = textstat.gunning_fog(text)

print('FOG index:',fog_index)

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

sentences = sent_tokenize(text)
total_words = 0
for sentence in sentences:
    words = word_tokenize(sentence)
    total_words += len(words)

avg_words_per_sentence = total_words / len(sentences)
print('avg_words_per_sentence:',avg_words_per_sentence)

import re


def count_complex_words(text, complex_words):
    # Split the text into words
    words = re.findall(r'\b\w+\b', text)

    # Count the number of complex words
    complex_word_count = sum(1 for word in words if word.lower() in complex_words)

    return complex_word_count
complex_words = ['analyze', 'sophisticated', 'articulate']

complex_word_count = count_complex_words(text, complex_words)
print('complex_word_count:',complex_word_count)

words = word_tokenize(text)
word_count = len(words)
print('word_count:',word_count)

import pyphen

# create a Pyphen object to hyphenate words
dic = pyphen.Pyphen(lang='en')


# split sentence into words
words = text.split()

# count syllables for each word
syllables_per_word = []
for word in words:
    syllables = dic.hyphenate(word)
    if syllables is not None:
        syllables_per_word.append(len(syllables.split("-")))


if syllables_per_word:
    avg_syllables_per_word = sum(syllables_per_word) / len(syllables_per_word)
else:
    avg_syllables_per_word = 0

print('avg_syllables_per_word:',avg_syllables_per_word)


import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


# Tokenize the text into words
words = word_tokenize(text)

# Tag each word with its part of speech
tagged_words = pos_tag(words)

# Find personal pronouns (tags starting with "PRP")
personal_pronouns = [word for word, tag in tagged_words if tag.startswith('PRP')]

# Count the number of personal pronouns
num_personal_pronouns = len(personal_pronouns)

# Print the results
print(f"Personal pronouns: {personal_pronouns}")
print(f"Number of personal pronouns: {num_personal_pronouns}")

