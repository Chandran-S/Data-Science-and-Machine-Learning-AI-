
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/impacts-of-covid-19-on-vegetable-vendors/'


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
text = ('''Impacts of COVID 19 on Vegetable Vendors
The COVID-19 pandemic has grown into one of the most major and socially disruptive health crises in recent memory, with growing worried about how the pandemic’s catastrophic economic and social repercussions are affecting food systems at both the global and local levels. Given the importance of the retail food environment in establishing and sustaining healthy diets, interruptions to specific aspects of it, such as the availability of fresh vegetables, could have a negative influence on population health, which has already been noted as a source of worry.
Despite the dominance of national and international supermarket chains, warehouse clubs, and supercenters in grocery retail, vegetables are offered in a number of other food retail settings. Fresh vegetable #vendors are typically smaller and more community-oriented than other restaurant or retail food outlets and can include chain or independent grocery stores, greengrocers, storefront stands, street carts, and even makeshift platforms dedicated to the sale of fresh vegetables. Smaller community fresh vegetable vendors who may conduct business on the sides of major streets or on storefronts have played an integral role in the food environment in large urban centers such as New York City (NYC), particularly in ethnic enclaves, despite the fact that fresh produce does not have a significantly higher nutritional value and fresh produce can also be purchased at these larger food retailers.
Since the COVID-19 pandemic began, many fresh fruit and vegetable vendors, notably street carts selling fresh vegetables in cities across the United States, including New York City, have been forced to close owing to a combination of falling demand and fear of catching COVID-19. The importance of fresh vegetable vendors varies by neighborhood within cities. Furthermore, these fresh vegetable vendors attract visitors and interborough shoppers from a variety of cultural backgrounds not only Asian ones who are looking for things that are not available elsewhere in the city or for the same low prices.
These fresh vegetable vendors, unlike larger, well-established grocery store vendors, may not have the financial infrastructure to sustain the shifts in supply and demand produced by the COVID-19 epidemic; consequently, the danger of closure or modifications in services may be greater for these vendors.
In order to assess the impact of the COVID-19 pandemic on services offered by fresh vegetable vendors, surveillance #data from both before and after the pandemic’s inception is required.
After a few days, vegetable dealers began venturing out without explicit permission and were quickly harassed by police. After a few weeks, the government relaxed the limitations, allowing vital traders to sell their wares (due in large part to the advocacy of vendor organizations and activist networks). However, the cost of doing business has increased dramatically, as vendors no longer have access to wholesale markets and suppliers, and they must spend more on travel expenditures owing to city-imposed travel limitations. Furthermore, with the partial lockdown still in place, the number of buyers has decreased, as have earnings. Perishable vegetables have a shorter shelf life in the summer heat, thus vendors are unable to capitalize on whatever produce they do have.
Consider the situation of Delhi at the starting of COVID 19. The state has launched an INR 5000 crore stimulus package for over 50 lakh vendors, realizing the serious consequences of their loss of livelihood. The targeted relief for vendors is a credit facility that will provide all sellers with an initial working capital of INR 10,000, but this will not be enough. Instead of credit, the government should consider changing it into a direct income benefit, such as a cash grant, to help people start earning money on a regular basis. The vendors require income support in order to resume work, and how will they repay the loan if they are unable to do so? Vendor organizations must step forth in the face of the ever-changing crises and lobby for vendors to be given the resources they need to continue their livelihoods. Vendor organizations could use the following as part of their advocacy agenda:
Finally, some fresh vegetable sellers may have shuttered for a period of time early in the epidemic, only to reopen recently. Alternatively, vendors may have launched soon after the in-person checks were completed, but still within the June-July 2020 endpoint timeframe. This is a drawback of the method; in order to offer the most reliable COVID-19 pandemic monitoring data, data must be collected in a short period of time.
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

