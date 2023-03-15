import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/impact-of-covid-19-coronavirus-on-the-global-economy/'


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
text = ('''Impact of newly discovered coronavirus on the Global Economy
What is Coronavirus? Is it a disease? How it
is spread to humans? How Coronavirus affects the world economy? Several
questions come in mind when we talk about the Coronavirus.
Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus. These are a type of virus that creates sickness in humans. There are many different kinds, and some cause disease. A newly identified type has caused a recent outbreak of respiratory illness now called COVID-19.
The COVID-19 virus spreads primarily through droplets of saliva or discharge from the nose when an infected person coughs or sneezes. Coronaviruses vary significantly in the risk factors. Some can kill more than 30% of those infected and some are relatively harmless, such as the common cold. Coronaviruses cause colds with major symptoms, such as fever, and a sore throat from swollen adenoids, occurring primarily in the winter and early spring seasons.
As it creates much chaos between humans and also affected the whole human diversity, it also creates a disturbance in the economy. The economy is the largest system of trade and industry by which the wealth of a country is made and used. If this economy will not work then the human race will also come to an end. So not only COVID-19 makes the humans sick but also it makes the economy a viral whole. 
The impact of coronavirus firstly affects Economic growth which is the factor that creates nation wealth. Economic growth creates wealth on a national scale for the government in the form of taxation, which is then redistributed accordingly to the services and communities that need it the most. Due to COVID-19 industrial production, sales and investment all fell in the first two months of the year which decline the whole economic growth. Such a slowdown in manufacturing industries has hurt countries that are interlinked to each other in terms of exports i.e. Asia Pacific economies, sub-Saharan economies and the USA economies. A reduction in global economic activity has lowered the demand for oil, taking oil prices to multi-year lows. This happens even before a disagreement on production cuts between OPEC and its allies caused the plunge in oil prices.
Job creation and employment opportunities are the second most factor which was affected by coronavirus as it often requires the skills and some wilful hands which necessitate in the long term of the business. Longer lasting and more intensive outbreak have half growth the opportunities in 2020 as factories suspend their activity and workers stay at home to try to avoid the virus.
Poverty is the third and most important factor which is diversely affected by COVID-19 as employment is not there and everyone was ordered to stay at home. When people will be in their own houses then automatically poverty will increase complimenting the decrease in economic growth.
There is also a big shift in stock markets, where shares in companies are bought and sold, affected many investments by the investors and shareholders or individual savings accounts. Central banks in many countries, including India, have slashed interest rates. Supermarkets and online delivery services also have huge growth in demand as customers stockpile goods such as hand sanitizers, toilet paper, rice, and orange juice etc.as the pandemic escalates.
At last the COVID-19 normally called coronavirus affects the world economy in a much-disintegrated manner. Some economies in which there are many deaths due to coronavirus will take more time to recover as their economy has gone before 2 years back. The economy can only be revived if the government makes more strict measures and coronavirus can be cured by the vaccine.
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


print('percentage_complex_words', percentage_complex_words(text))  # Output: 60.0

from textstat import textstat

fog_index = textstat.gunning_fog(text)

print('FOG index:', fog_index)

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

sentences = sent_tokenize(text)
total_words = 0
for sentence in sentences:
    words = word_tokenize(sentence)
    total_words += len(words)

avg_words_per_sentence = total_words / len(sentences)
print('avg_words_per_sentence:', avg_words_per_sentence)

import re


def count_complex_words(text, complex_words):
    # Split the text into words
    words = re.findall(r'\b\w+\b', text)

    # Count the number of complex words
    complex_word_count = sum(1 for word in words if word.lower() in complex_words)

    return complex_word_count


complex_words = ['analyze', 'sophisticated', 'articulate']

complex_word_count = count_complex_words(text, complex_words)
print('complex_word_count:', complex_word_count)

words = word_tokenize(text)
word_count = len(words)
print('word_count:', word_count)

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

print('avg_syllables_per_word:', avg_syllables_per_word)

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

