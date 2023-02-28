
import requests
from bs4 import BeautifulSoup

# Example URL
url = "https://insights.blackcoffer.com/ai-tool-alexa-google-assistant-finance-banking-tool-future/"
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
text =('How machine learning used in finance and banking?'
'Through AI tools like natural language processing, Alexa and google assistant has led the retail industry in its rise towards conversational commerce. As if a customer was interacting with a clerk in a retail store, conversational commerce makes it possible for users to engage with software to research, purchase, or get customer assistance with products and services across a wide range of industries. With Alexa, for example, users can ask any Alexa-enabled device to add an item to an Amazon shopping cart, set a purchasing reminder when a product is running low, or carry out a complete purchase without having to access a shopping cart. The result is a seamless conversational experience that enables consumers to carry out transactions as quickly as it takes to speak a sentence.'
'Through AI tools like natural language processing, Alexa has led the retail industry in its rise towards conversational commerce. As if a customer was interacting with a clerk in a retail store, conversational commerce makes it possible for users to engage with software to research, purchase, or get customer assistance with products and services across a wide range of industries.'
'With the advent of personalized products and on-call delivery, customers have come to expect a new standard experience: fast, easy, accurate, and personalized. Accomplishing this without sacrificing your workday can be a challenge, since the data processing required to meet these needs is immense. Luckily, virtual agents (VAs), powered by conversational AI, can utilize this information faster and more accurately than humans, finding insights and automating communication to deliver an enriched customer experience. If you invest based on these improvements, you’ll find that implementing these tools delivers a powerful competitive advantage. AI has helped in automobile, education, retail and commerce, finance and banking and healthcare.'
'Voice AI has powered the wheels of conversational e-commerce, which has impacted the way the customer communicates with the brand in multiple industries. Brands generally build a campaign to emotionally connect with customers, for long-term growth. With Voice, brand campaigns need to be short and ones that can lead to immediate buying. Conversational e-commerce is still in its nascent stage and it is expected to grow manifold in the coming years. The future of shopping is going to Voice AI and marketers have to get on the bandwagon fast to increase their brand value and visibility. Targeting will have to be highly personalized for success.'
'Despite its narrow focus, conversation AI is an extremely lucrative technology for enterprises, helping businesses more profitable. While an AI chatbot is the most popular form of conversational AI, there are still many other use cases across the enterprise. '
'While an exclusively chat- or voice-based shopping experience for all scenarios may never completely replace the in-person experience, conversational commerce will continue to grow as an added method of convenient and efficient communication. As users continue to become more accustomed to engaging with chatbots and voice-driven interfaces, expect more innovations in the space as brands continue to develop their unique conversation-based solutions.')

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

