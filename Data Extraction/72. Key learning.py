
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/lessons-from-the-past-some-key-learnings-relevant-to-the-coronavirus-crisis/'


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
text = ('''Lessons from the past: Some key learnings relevant to the coronavirus crisis
So, not beginning with once upon a time because it is not a fairytale. Before the advent of this deadly virus called Coronavirus or with all love that we might want to call it COVID-19, the life wasn’t smooth as people are cribbing about.
It is written by Adam Smith that Human wants are endless and thus we know humans can never be satisfied. When we were busy and all the work and stressful lives, we all wanted breaks, we used t crib and cry about not getting time for family and friends. Not having time for self
This Coronavirus outbreak had taught each one of us that :
How important it is to be thankful for everything we have in our lives while sitting I just realize that how lucky we are that we do not have to go out amidst this crisis to work and earn bread. How sorry I’m for each the daily wage workers who have no other way but to go out
The second lesson is that we all can survive without fast food, we all are chefs and have made a lot of appealing and tasty dishes without spending a huge and bogus amount.
The third lesson is family is everything, during this quarantine I realized that spending time with my family is so stress-busting and helps me be creative.
We need to have other hobbies than going out and chilling.
We need to look at the weather, it’s April and it is not that scorching heat we used to face, it’s lovely, nature is recovering. Whenever I wake up in the morning I go to my terrace for Yoga and find many families there, walking, gossiping, it has brought everyone close. Mended relation.
All the Instagram stories make me realize that we all have talent just we don’t have time to self introspect, we all are running in a race to be the best.
From an economic point of view, though savings are leakage in the money flow but still savings are important for unforeseen circumstances.
This lockdown has given us all time to reconnect and get together with all our loved ones, to do what we want.
The main lesson is that we should enjoy life and every moment as it is. We never know what might happen to anyone. We never know who could be the last person we are talking to, last person we text, let us all be nice to each other
There are many lessons that we have learned. One of them is the awareness about the cleanliness. The term CLEANLINESS, that took Modi Ji about 5 years to teach people about Swachh Bharat Abhiyan, a single case of the virus has taught everyone the importance of cleanliness and hygiene. Corona Virus crisis has taught how the humans at halt can lead the nature to work at itself.
Nature has its own mysteries and how what the humans have done to its beings is done to them, they are too imprisoned like the animals they have had imprisoned for years away from their homelands.
This Crisis has gaped the differences between the communities and brought the Humans together.
We can see a wave of excitement in everyone when Modi Ji gives his tasks. Still, remember how my colony just rang up with the sound of bells and plates and claps. And how beautiful it looked with the balconies lit up with Diyas and Candles. It looked like Diwali in the month of April but Diwali with no pollution and noise.
This crisis has made everyone of us thankful to the medical and nursing staff who had been working 24×7 hours , away from their families , so that we can stay safe with our families .
It has just made me realize that we never know , if we’ll see the sun tomorrow , If we’ll see the people we love , if we can talk to them again , so let’s not hold grudges and let the go and flow .
Corona Virus has united all the humanity against a cause to defeat a virus, this virus has made the world a better place, there are no terror attacks, there are no rape cases, there are no murders, no loots. The Earth is all healing.
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

