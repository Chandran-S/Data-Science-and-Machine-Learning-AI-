
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/what-is-the-repercussion-of-the-environment-due-to-the-covid-19-pandemic-situation/'


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
text = ('''What is the repercussion of the environment due to the COVID-19 pandemic situation?
What is COVID 19 pandemic?
On 31st December 2019, a novel coronavirus was identified in Wuhan, China. It spread like wildfire in the world. On 11th March 2020, WHO declared COVID 19 as a pandemic and issued guidelines like Sanitizing hands regularly, wearing masks, and social distancing as viable methods for prevention against this virus.
What is social distancing and how is it helpful?
Social distancing means staying away from other people. So, to limit human interaction various governments all over the world chose life before the economy and ordered complete lockdown in their countries. All schools, colleges, offices, factories, public places, etc. were shutdown from immediate effect.
As of 7th June 2020, COVID 19 confirmed cases worldwide are 69,94,605 with 4,02,453 deaths and 34,20,048 recovered.
The USA didn’t impose lockdown in the early stages due to which it has nearly 19,88,545 confirmed cases with a death rate of 5.6% (as of 7th June 2020) whereas in India lockdown was imposed at the first stage, and so we have 2,47,000 confirmed cases with a death rate of 2.8%.
Impact on Environment
Rights reserved to Statista
As we know it is us, Humans, who keep exploiting nature which has nurtured us and turn a blind eye to the havoc of our misdeeds. Current environmental concerns like depletion of the ozone layer, climate change, soil erosion, air pollution, water pollution, soil pollution, acid rain, noise pollution, loss of biodiversity, and many more are a result of our selfish and never-ending desires for development at the cost of the environment.
But when the lockdown was imposed worldwide, our lives just stopped. Air traffic (due to flights) dropped by 95% (Cntraveller). Since no one was allowed to leave homes, Carbon footprint per person dropped significantly. The Hindu newspaper reported a 17% decline in carbon pollution during the pandemic peak. According to an article in Economic Times onMay 14th,2020, Global air quality has improved this year due to lockdown.
Economic Times posted an article on 8th April 2020 titled “COVID 19: All world’s now a zoo, only this time with animals on outside and humans on inside”.
This sounds like lock down is good for Environment . Right?
Humans are getting what they deserved, animals are safe and the environment is healing due to a reduction in pollution levels.
But, NO.
All this is just the tip of the iceberg .
The reality is that humans, despite being under threat due to COVID 19, cyclones, earthquakes, and locusts attacks are still damaging the environment.
According to Voicesofyouth.org:” There has also been an increase in medical waste – much of the personal protective equipment that healthcare professionals are using can only be worn once before being disposed of. Hospitals in Wuhan, for example, produced over 200 tons of waste per day during the peak of their outbreak, compared to an average of fewer than 50 tons prior. “
According to Mongabay News: “Despite COVID, Amazon Deforestation is racing higher. The new figures come amid rising fears that illegal loggers and speculators are using the COVID-19 crisis as an opportunity to invade indigenous lands and protected areas in Brazil”.
According to an article in Conservation International , an expert said ” In Africa, there has been an alarming increase in bush meat harvest and wildlife trafficking that is directly linked to COVID-19-related lockdowns, decreased food availability and damaged economies as a result of tourism collapses”
Apart from the above-mentioned media channels, various other news articles have reported such incidents from various locations around the world.
Now, what do you think :
COVID 19 has helped the environment for good or just deteriorated it further? Check for yourself.
Follow social distancing guidelines . Stay Safe .
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

