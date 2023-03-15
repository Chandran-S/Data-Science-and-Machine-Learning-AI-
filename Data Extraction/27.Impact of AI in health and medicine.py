
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/impact-of-ai-in-health-and-medicine/'

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
text = ('''Impact of AI in health and medicine
AI allows those in training to go through naturalistic simulations in a way that simple computer-driven algorithms cannot. The advent of natural speech and the ability of an AI  computer to draw instantly on a large database of scenarios means the response to questions, decisions, or advice from a trainee can challenge in a way that humans cannot in health and medicine
Wearable health trackers-like those from FitBit, Apple, Garmin, and others- monitor health rate and activity levels. They can send alerts to the user to get more exercise and can share this information with doctors.
Machine learning algorithms can process unimaginable amounts of information in the blink of an eye and provide more precision than humans in spotting even the smallest detail in medical imaging.  A few of them are Blackford, Zebra, Enlitic, Lunit.
The company “Zebra Medical Vision” developed a new platform called profound, which analyzes all types of medical imaging reports that are able to find every sign of potential conditions such as osteoporosis, aortic aneurysms, and many more with a 90% accuracy rate.
For eg, the digital health firm #HealthTap developed “Dr. AI”, and apps like Babylon in the UK use AI to give medical consultations based on personal medical history and common medical knowledge. Users report their symptoms into the app, which uses speech recognition to compare against a database of illness and asks patients to specify symptoms to triage whether they should go to the ED, urgent care, or a primary care doctor.
AI robot-assisted surgery: Robots have been used in medicine for more than 30 years. Surgical robots can either aid a human surgeon or execute operations by themselves. They’re also used in hospitals and labs for repetitive tasks, in rehabilitation, physical therapy, and in support of those with long-term conditions.     
 Health chatbots such as Babylon, Ada, and mostly close-ended communications.
#MICROSOFT: Predictive analysis in vision care
#GOOGLE: clinical decision support in breast cancer diagnosis
#IBM WATSON: precision medicine in population health management
Actionable medical insights:  An ever-increasing amount of medical data are being digitized at all public and private healthcare institutions. However, by its very nature, this kind of data is messy and unstructured. Unlike other types of business data, where traditional statistical methods can be used for quick insights, patient data is not particularly amenable to simple modeling and analytics tools.
 For eg.- Enlitic, a San Francisco-based start-up, has a mission of mixing intelligence with empathy and leverage the power of AI in health and medicine for precisely generating.
Therefore, a massive parallel effort to rationalize the legal and policy-making is needed to bring the full benefit of advancement in AI technologies into the healthcare space. As technologies and AI/ML enthusiasts, we can only hope for such a bright future where the power of this intelligence.
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

