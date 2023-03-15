
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/management-challenges-for-future-digitalization-of-healthcare-services/'
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
text = ('''EManagement challenges for future digitalization of healthcare services
Management acts as a guide to a group of people working in the organization and coordinating their efforts, towards the attainment of the common objective.Management challenges for future digitalization of healthcare services
    In the current day and age, a wave of digitization has taken over the world. All the emerging technologies like Artificial Intelligence and others are helping people live better and easier life. The service sector has also benefited a lot from digitization. It got further boost when Prime Minister Narendra Modi, in the year 2015, launched a campaign known as ‘Digital India.’ Among the various industries in the service sector, digitization has had a massive impact on the operation of the healthcare and diagnostic industry. It has helped in the development of this industry and enhances life for millions of people. The condition of India in terms of healthcare has been quite grim. Patients who need attention are neglected and are unable to avail proper treatment or diagnosis. However, with the coming up of digitization, there has been hoped for this industry also.
According to Dr. Keshab Panda, CEO & MD, L&T Technology Services,
This is a model of healthcare where doctors and hospitals are paid based on patient health outcomes. The coming up of digital tools in this segment of healthcare can be considered as the starting tool. Dr. Panda further says that the implementation of advanced digital technologies in patient-care domains like- mobile health apps, telehealth, wearables, and remote monitoring can help in enhancing accessibility, boost efficiency and augment the effectiveness of treatment and preventive care.
The healthcare industry over the past few decades has changed drastically. With the coming of emerging technologies, new surgical procedures and medical devices have been made available.
While talking about connectivity in healthcare, Dr. Panda said that with digitization taking over the industry, it is only a matter of time when with the help of the internet and smartphones, digital healthcare will be everywhere. The internet for once has helped doctors connect to their patients and also with one another. This has helped in enhancing the doctor-patient engagement by increasing their interaction time.
Big data aggregates information about a business through formats such as social media, eCommerce, online transactions, and financial transactions, and identifies patterns and trends for future use.
Although ransomware, data breaches, and other cybersecurity concerns are nothing new to the healthcare industry, the 2020 Covid-19 pandemic revealed just how vulnerable sensitive patient health information really is.
The recent growth of digital health initiatives- like telehealth doctor visits — is a major contributor to the severe increase in breached patient records. As more healthcare functions continue to move online over the next year, it’s extremely important to ensure these processes are protected from outside threats.
Medical practices are citing patient collections as their top revenue cycle management struggle as patients are becoming responsible for a larger portion of their medical bills. In order to help encourage patients to submit payments in a timely manner, providers must adhere to patient payment preferences.
To meet patient expectations and improve the user experience, ensure billing statements are patient-friendly. You should offer paperless statements and a variety of payment options (e.g. credit card, etc.) via an online patient portal and utilize the latest payment technologies, such as mobile and text-to-pay. New features like text or email reminders help effectively communicate with patients and encourage them to pay their financial obligations.
The medical insurance landscape has experienced some significant changes in recent years. As more patients are responsible for a larger portion of their healthcare bill, they naturally demand better services from their providers.
Healthcare organizations will face tougher competition in attracting and retaining patients who demand an experience that matches the level of customer service they expect from other consumer brands.
They demand a streamlined patient experience so they can “self-service” to resolve most questions, issues, or concerns (e.g., downloading an immunization record, booking an appointment, paying their bills, or checking their account/insurance status) whenever, wherever, and however is most convenient for them.
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

