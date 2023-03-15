
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/coronavirus-impact-on-the-hospitality-industry-3/'

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
text = ('''Coronavirus: Effect on the Hospitality Industry
The hospitality sector in India has witnessed unprecedented deceleration this year in the wake of the COVID-19 pandemic. The Indian hotel sector has been hit hard, tussling with significantly low demand, with very few future bookings. Since the outbreak of COVID-19, companies have been canceling interviews at the top hotel management colleges in India. The overall hiring sentiment in the country is witnessing a negative impact in the short term with 60-65 percent of interviews getting postponed, especially in the hospitality services sector, following the slowdown across industries triggered by the coronavirus pandemic. Even the students who were aspiring for a career in the hospitality industry are on the verge of full swing. Currently, with the imposition of Section 144, there are barely any bookings being made for the future, and the current ones all stand canceled. In this scenario, there is limited scope for quick revival.
It’s proud to see that this sector shows its maturity level: in working together, showing their true hospitality commitments in helping out our society where they can. For example by making their venue available for hospital beds and hospital employees. The situation we are in also brings new business models and opportunities, in defining, for instance, new delivery concepts, human capital sharing platforms, initiatives in promoting the “staycation or holiday concept” and the use of the less productive time to work on activities that were normally pushed forward like asset counts, security plans, defining standard operating procedures, social media plans, etc.
This may seem obvious, but it’s worth looking at this a bit closer. The main reason there was such widespread panic was that the government announced its advice to avoid public meeting places like restaurants, bars, and pubs without simultaneously announcing a complete ban or a plan to support the businesses whose revenue streams they’d just denied. Many of these businesses operate on low margins and very fragile cash flow already so the prospect of survival for an indefinite period of time without income seemed very bleak indeed. This has taken a mental toll on already stressed business owners and their worried staff.
The impact of the novel coronavirus on India’s hospitality sector jobs is nothing short of severe. While most economists expect things to rebound in the latter half of the year, uncertainty still lurks. A sheer job trauma is staring at the hospitality industry in the near future. This is as the chances of losing a job are at high risk. Furthermore the entry-level has postponed the hiring for the near future. Let us check here some of the facts concerning the drop in the various hotel occupancy sectors.
The hospitality industry can use this chasm to prepare for the upcoming demand by focusing on marketing and up-gradation. This ‘Stop Gap Plan’ is about maintaining a thread of communication, using social media and advertisements, with the consumers. It’s also about strengthening the communication within the company, making a budget and plan for re-opening, and utilizing this period to fix and upgrade whatever is possible. Another step is where all the action is required by the sector. Once the outbreak of the virus is contained and the world is set to travel again, it is suggested that any plan of re-opening must be done keeping long-term benefits and safety compliances in mind. It is imperative that hospitality companies reach out to deferred and canceled bookings and give due attention to domestic travelers. The hotels and airlines must slowly roll out their services rather than starting everything instantly and not get caught up in spending.This way Coronavirus Effect on the Hospitality
The government is already taking measures to combat the effects of the pandemic on the country. Here are the multiple measures, if taken by the government of India, can help the Indian hotel industry weather the current storm.
Really this is anyone’s guess. We don’t yet know how long it will take for the epidemic to peak and for us to be allowed out again, but one thing I am sure of is that we’ll all be desperate to meet up with friends over dinner or have a pint in the pub! With proper support from the government I think the hospitality industry stands a good chance of bouncing back quickly, but perhaps not in its former form. It is likely that we will see a kind of natural selection take place among restaurants, leaving space for innovative new entrants. The crisis has already forced several restaurants to innovate by developing new services and perhaps, if regulation allows, these will stick, becoming new models for restaurants in the future and offering them a more diverse and resilient set of revenue streams. Thus Coronavirus Effect on the Hospitality and other sectors.
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

