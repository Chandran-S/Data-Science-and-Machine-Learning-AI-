
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/coronavirus-impact-on-the-hospitality-industry/'


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
text = ('''Coronavirus: Impact on the Hospitality Industry
Before jumping on the topic I would like to give an overview of what is Coronavirus, Covid-19, how it spreads, and its symptoms.
Waitresses, hotel housekeepers, and casino dealers are among the more than 15 million hospitality jobs in U.S. cities at risk from restrictions being put in place to deal with the spread of Covid-19.  Bureau of Labor Statistics data through May 2018 covering 40 occupations critical to America’s hospitality and gaming industries.
Hospitality workers as a share of each metro area’s workforce (circles sized by total hospitality jobs).
As of 20 April 2020 at least 23 states have closed bars or restaurants, around 20 have prohibited gatherings of more than 50 people, and several have begun implementing curfews and shuttering non-essential businesses. The most extreme measures have occurred in California where a shelter-in-place directive was announced by Gov. Gavin Newsom on Thursday. The order requires 40 million California residents to remain at home except for essential activities or jobs. In addition to critical government and healthcare roles, cafes and restaurants are allowed to stay open but only for take-out or delivery, which has already led to mass layoffs.
Nowhere are there more threatened jobs than in the New York metro area, where one million people work in hospitality. This includes 157,000 waiters and waitresses, 40,000 bartenders, and 8,500 hotel desk clerks. On 20 April 2020, Gov. Andrew Cuomo ordered all non-essential workers to remain at home for the foreseeable future. The Los Angeles area has the second-most such workers—around 800,000—including 22,000 people who work at amusement parks and recreation facilities. This represents between 11% to 13% of these cities’ respective workforces in recent years.
Most Hospitality Jobs
There are 4.6 million hospitality workers in the top city clusters
The pain of an extensive and prolonged coronavirus-related shutdown will be especially felt in the nation’s tourism hotspots. Roughly one in four workers in beach destinations like Kahului, on the island of Maui in Hawaii, and Myrtle Beach, South Carolina are employed in the hospitality sector. The same goes for gambling towns like Atlantic City and Las Vegas, where the governor of Nevada recently announced a 30-day shutdown of all casinos.
The hospitality industry matters most in beach and gambling destinations
All but five of the 40 occupations in this analysis fall into broad categories that were less likely than the workforce overall to work from home on average in 2018, according to the BLS American Time Use Survey. Additionally, 17 of the roles, including restaurant wait staff and table game dealers, require at least arm’s-length contact with others, based on physical proximity scores compiled by the O*NET database of occupational information. At a time when most major cities are urging residents to stay at home and practice social distancing, few jobs are more at risk than these.
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

