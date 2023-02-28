
import requests
from bs4 import BeautifulSoup

# Example URL
url = "https://insights.blackcoffer.com/all-you-need-to-know-about-online-marketing/"
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
text = ('All you need to know about online marketing'



'Ever wondered how you get notified of the products or services you want or you have been looking for for a long time. Now how does this happen? Let me start with the simple and the most known fact- Marketing,'
'Marketing is a common term which everyone knows and is aware of. Marketing is the action of promoting products and services, including market research and advertising.'
'Traditional Marketing was working fine for all these years. So why there was a need for online marketing?'
'The story begins from the internet era, The online presence of customers-'
'There are 4.72 billion people are on internet users in the world today and the number is increasing day by day. So if the companies want to create awareness about their products and services their is a huge audience present online.'
'There are a lot of benefits of online marketing like a large audience, benefits of targeting on the basis of demographics, location, age, gender, and many more. Because of this diversity, almost every type of company can use the Internet to reach any audience. All would find something of their liking.'
'Let’s take a look at what does online marketing involves.'
'These days, almost everybody is on social media. The majority of people use Facebook, Twitter, Instagram, and other social media platforms to communicate with their friends and relatives. Some people have created companies solely based on their social media activity.'
'You can, however, promote your Knowledge Commerce products through social media. Whether or not you advertise on these sites.'
'The best social media networks for advertising-'
'Facebook advertising'
'Instagram advertising'
'Twitter advertising'
'Pinterest advertising'
'LinkedIn advertising'
'Snapchat advertising'
'Content marketing is a strategic marketing strategy that focuses on producing and delivering useful, appropriate, and reliable content in order to attract and maintain a specific audience — and, eventually, to drive profitable consumer action.'
'In simple words, content marketing is a marketing strategy that producers and delivers relevant and reliable content to attract potential clients and to also retain existing clients.'
'About half of all website traffic originates from a search engine. People who use searches are often high-intent buyers. This indicates that they are searching for a particular item. They’re all set to buy the products and services.'
'The majority of online marketing is focused on pay-per-click (PPC). However, when you hear the word “PPC,” it refers to search ads.'
'Pay-per-click (PPC) advertising on search engines, social media sites, and other online venues can be extremely successful.'
'So the next time when you search for a product and services and you find similar ads on the internet this is a kind of internet marketing.'
'Email marketing is the highly successful digital marketing technique of sending emails to prospects and consumers. It allows communicating directly with the present, former, and potential customers. businesses will inform the audience about new products as they become aware of them.'
'Banner ads are rectangular or square advertisements that appear above, in the sidebar, or below the content on websites. Usually, a banner ad leads to a sales or landing page. You will get great results by running banner ads on websites that attract members of the target audience.'
'An affiliate marketer, like a car salesperson, only gets paid when someone buys the stuff. You may not have to pay if there are no purchases.'
'Affiliates are free to sell your goods anywhere they choose (as long as the material follows the terms of service of the website). It’s a fantastic way to reach new markets.'
'Anyone can start their affiliate marketing career by registering on the different affiliate websites.'
'Influencer marketing is a new trend for online marketing. They have a strong following, can inspire people to buy your goods, and are loved by their viewers.'
'The type of online marketing that will work best for the company will be determined by many factors, including the nature of your industry, the tastes and demographics of the target market, and budget. Market analysis will guide to the best strategy or combination of strategies for your offerings, and comprehensive performance metrics will show you which are the most effective.'
'Online marketing is a rapidly expanding industry that benefits companies in a variety of ways. The number of people who purchase goods and services online is on the rise. As a result, an increasing number of businesses around the world are turning to internet marketing to communicate with consumers and advertise their goods and services.')

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

