
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/how-will-covid-19-affect-the-world-of-work-4/'

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
text = ('''How will COVID-19 affect the world of work?
COVID19 affects the world from every way life emotional to social, philosophy to physical. Economical to the thread of life…it is known as present as part of life. which tell as on a daily base to the world just stop your self where you are present..if you move you will lose to yourself and to others, it’s like a story which tells which we listen in our childhood one dirty fish harm the all see.
So now we have one only options to not move which tell as that we can’t work as we working 3months earlier….till the hope of living make new invention to challenge the world biggest challenge…upto that time we have stop to as to work as we are working eailer that mean we have to do essential work…this self defince throory and sanjvini of life is going to affect world of work as i mantion in many why before we understand current sanrio…i wold like to share some personal life experiences with you i belong to small beautiful city it far from 250 km from india capital new Delhi…story of today my city is not that mctah complex as big metro city or world most developed nations are facings problem of covid but still we are at self isolated face as city ti safe as here the problem which tell as beautiful stories 4week earlier when there no lock down was there our maid came to work at home and my mom asked here to stop working she repided very beautiful that if we stop working who will going to take care of my family. here directly saying moany is one thing at that time but may mon replied here that we will help you and I also replied here that government will also help if their situation will be critical but she reply again and told me that what use of here good health that if we can’t use to for work and she continues to come on work till 2day of 1st lockdown. She stops till when law forced here to stop here…my mom asked here yesterday when she comes to our medical shop to take medicine for her family. What you are doing know days she replied again in a beautiful way and told us that she said I going into gurdwara to for making food for a needed person with taking some sought of privation I asked why you are doing she replied that I am financial strong so that I can help other but I can do which I can do is work for other…this story give many views regarding the current working scenario. even I was thinking earlier we are going fave of life where we going to so lazy and more addicted of device yes we going to attend of these things but as our DNA and value system and living force to earn money for living or living for money is going to work as past this story till me that we going to past as future today is just phase which will pass out one day with some spreading difficulty and learning of life that give a clear picture that we not going to time world work stop in future and if we talk about the current scenario we still not our critical work yes we have to stop or we many cases we stop using unnecessary think. let discuss the current world work situation with some facts and figures…..In the worst-case scenario, the world economy could contract by 0.9 percent in 2020,” the DESA said, adding that the world economy had contracted by 1.7 percent during the global financial crisis in 2009. this will result in a massive increase in the unemployment rate even some figure start to come regarding jobless like:
America till know project that they are going to be 2 crore jobless due to covid19 if this will continue till 3 moth more, see face developing nation when the name of doping nation come to the first ma,e come into mind is India
with reference of business stand article;
we estimated the cost of lockdown nearly 17.008 lakh crore or rupee 17 trillion dollar loss to a different sector of business if strictly lockdown is followed till 17 may let’s see sector ways…
the mining sector is going to affect completely with the loss of 0.6975 lakh crore rupees only and manufacturing sector going face loss of 4.86 lakh crore only, construction cooling shut down for 2.385 lakh crore loss, trade hotel transport business  and broadcasting facing loss 5.445 lakh crore, financial real estate are going face loss of6.3225 lakh crore which make total 17 lakh crore loss 
this finger is eagle to the total focal budget of the movement of India of 2015 and India estimate COVID 19 costs the 3 crore people jobless in India so imagine where we are going but still, we have hope to get well soon because we know when to work or when to stop us. My last opinion state work of word today affected but after some time we will at the same path where we in the past it just need some afford.
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

