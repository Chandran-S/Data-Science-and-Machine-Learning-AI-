
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/how-covid-19-is-impacting-payment-preferences/'

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
text = ('''How COVID-19 is impacting payment preferences?
I would rather pay cash – Before COVID-19.
I would rather make online payment – After lockdown.
During this lockdown, one can observe a number of small positive changes in our surroundings. One such positive change is using online mode of payment even if they are small in amount as it counts as a big step towards DIGITAL INDIA and self-development as well.
According to Economic Times, 42% of Indians say that they have started using online mode of payment. Some small tasks like mobile phone recharge, bill payments, buying groceries, etc., are some essential tasks that cannot be ignored, and making an online payment is way too convenient for them. Also, multiple schemes have been initiated by the government to promote online payments like Lucky Grahak Yojana and Digi Dhan Vyapari Yojana. The details of the same are as follows:
LUCKY GARHAK YOJANA
DIGI DHAN VYAPARI
The frequently used apps by the consumers and the market share of apps are as follows:
Paytm- 33%
Google Pay- 14%
PhonePe- 4%
Amazon Pay- 10%
BHIM- 6%
Other Apps- 33%
Not only the sellers but also the buyers have taken a step forward in making online payments and have been exploring the various offers and discounts. The sellers have started their own online shops and made online payment mandatory in highly contaminated zones. The buyer has also taken advantage of home deliveries to follow social distancing. To add to the advantage of buyers and sellers KYC is no more a necessity to make money transfers.
There has been an increase in the services provided by banking sectors as well. The FM Nirmala Sitharaman has provided the directions to not charge for cash withdrawals from ATM’s. Various services like sending and receiving money, blocking cards, credit card payments, credit card pins, etc., are being provided online.
E-payment options like Paytm, etc., have been open up for investing money online in stocks, insurance, etc. Even online shopping apps like Amazon have come up with various offers for their in-built wallets.
Emerging technologies and high competition among online payment apps are providing benefits to buyers and sellers as well. Sellers have been able to increase their reach with online shops and online payment stands as a guarantee for them. Buyers have been in the most advantageous position ever after the digital payment has come into the market. They are being provided with a number of choices to choose from as well as make instant payments for offers. This all sums up the likeliness of digital payment in the market. So yes, we can say, YOU (DIGITAL PAYMENT) HAVE A BRILLANT FUTURE, CHILD!
We all have heard this statement “Precaution is better than cure” and here is the time to follow the same. It has been proven as a fact that the coronavirus can persist on paper for four or five days. That means the coronavirus can persist on the paper currency for four or five days. Thus, it is important to avoid the usage of paper currency.
A revolution will take place only when the below poverty population of India will start using the online mode of payment and will be literate enough to not get caught in any scam.
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

