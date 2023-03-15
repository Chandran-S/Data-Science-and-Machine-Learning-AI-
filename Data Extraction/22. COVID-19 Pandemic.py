
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/how-data-analytics-and-ai-are-used-to-halt-the-covid-19-pandemic/'
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
text = ('''How Data Analytics and AI are used to halt the COVID-19 Pandemic?
Even though COVID-19 has not yet halted and we are facing the nth wave of the coronavirus outbreak across several countries, most notably the US, India, and Brazil. It is a fact that Data Analytics and AI are the big guns of our artillery in this fight against the COVID-19 pandemic. It has helped us in several stages of this outbreak, like the detection of its first outbreak, vaccine development and manufacturing contact tracing, and future hotspot detection. Some of these interesting applications are discussed in this article.
A lesser-known fact is that the COVID-19 outbreak was first detected in Toronto, Canada, nearly 7,230 miles away from the first outbreak, nine days before the WHO issued its warning. It was with the help of Big Data Analytics and AI, more specifically Deep Learnings (DL, a subset of Machine Learning) application in Natural Language Processing (NLP) to analyze text inputs that traced the surge of pneumonia cases in the Wuhan province of China. The specialty of DL algorithms is that they mimic the brain cells called neurons and can identify patterns in Big Data. This DL-backed software is used as inputs, reports from public health organizations, global airline ticketing data, etc. These were used to flag unusual surges and potential spreads of infectious diseases.
The next application of Big Data Analytics and AI was in the Research and Development of drugs to halt COVID-19. AI was used to analyze the protein structure of the virus, findings that were significant in the progress of vaccine development. In preliminary studies, it was found that it does not mutate as fast as other viruses such as HIV, which means that a prophylactic vaccine is a better way to proceed rather than a therapy. But there is also some evidence supporting the fact that when we find any kind of cure for it, there is a chance of the virus mutating, which is what happened and major mutations have been found in the UK, Brazil, and South Africa. AI also assisted scientists in rapidly shortlisting a set of already available vaccines that could be effective against the coronavirus.
Another interesting application of AI can be found in the selection of the right candidates, i.e. most likely to test positive for testing coronavirus in case of insufficient testing resources. This method was first exercised on Greek borders and was called project EVA. Whenever a traveler wanted to come into Greece, he had to fill out a form known as Passenger Locator Form (PLF) at least 24 hours prior to arrival, containing information on their origin country, demographics, point, and date of entry, and the intended destination. EVA then allocated testing resources according to the size of the set of passengers to be tested. After the test results, if found positive, they are put in quarantine. The results were sent back to the program for real-time learning.
The question remains how EVA made allocations, It was found that, statistically, only the origin country and the city were significant factors for screening. Ultimately, from a variety of countries and city pairs, EVA had to predict how many testing resources were to be allocated at each entry point and to particular passengers from a location is technically called the Multi-Armed Bandit (MAB) problem, and the chosen method to solve this problem was an AI algorithm called optimistic Gittins index. This algorithm identified on average 1.85x as many asymptomatic, infected travelers as random surveillance testing, and up to 2-4x as many during peak travel. After the test results, if found positive, they are put in quarantine. Following the collection of significant data through the aforementioned process, after a certain period, policies were made categorizing them separately and imposing restrictions on travelers from the specific location. This EVA as presented above was in operation from August 6th to November 1st processing around 38,500 PLFs each day and testing on an average 18.5% of households entering the country every day.
Above mentioned applications just show the tip of the iceberg and there is more to get into some of the other developments to watch for include the use of Image Recognition to identify covid based on x-ray images, the use of Deep Learning to predict the 3-D protein structure associated with COVID-19 and so on.
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

