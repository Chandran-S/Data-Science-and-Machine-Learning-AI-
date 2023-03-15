
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/what-is-the-future-of-mobile-apps/'


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
text = ('''What is the future of mobile apps?
From what I have learned on the mobile apps development market recently, it is becoming harder for an Android or iOS developer to find a suitable and well-paying job in the market.
Some major reasons include the increasing supply of mobile engineers, people generally stop downloading apps, and many more.
In this article, I will go through a more detailed explanation of the recent trend mentioned above and my two cents on what mobile engineers could do to prepare themself for the upcoming challenge! and the future of mobile apps.
Although the demand for mobile developers has not fallen sharply compared with previous years, the growth rate has slowed. On the other hand, owing to a great number of coding boot camps that target to bring up mobile and frontend engineers, a great number of developers are flooding into the developer job market.
The supply side for mobile engineers has increased rapidly in recent years, raising the standards for qualified mobile engineers. The days of a mobile engineer easily getting over 10 job offers are over.
Recode ran an article in mid-2016, that begins: “The mobile app boom kicked off in July 2008, when Apple introduced the App Store. Now it is over.”
According to data gathered and analyzed by the CakeResume team, JavaScript and Python are now the most in-demand programming languages for companies who post product development jobs on CakeResume, both together taking up almost half of the job opportunities, while mobile engineers taking up merely 10% of the job opportunities.
While App usage continues to grow and revenue also ascends, the majority of consumers actually download zero apps per month
let’s face the truth, born and raised up in the era of the information explosion, individuals don’t have the leisure to check on what’s new on the app store every day. When it comes to choosing an App to download, many people feel overwhelmed by the sheer number of options available.
Take the productivity category of the App Store as an example, there are over thousands of Apps in this category, and new Apps popping up almost every week. It is almost impossible for users to dig through and learn all the apps. They just choose what is on top then click Install.
And not to mention that if you could build an awesome productivity app, Google and Apple could do it 10 times better and faster than you. In the end, there’s not as much need for individual apps to accomplish similar convenience factors.
Although the bull run for mobile development has ended, there are still over 2 billion smartphone users, 27.5 billion mobile apps downloaded, and time spent per day on mobile devices has increased rapidly in recent years. Moreover, apps offer a user experience that even ‘Responsive Websites’ are unable to provide.
In order to stay on top of the game, you will have to differentiate yourself from the pack — other mobile engineers. If you are a mobile engineer who only knows how to code and tweak your App’s UI, you aren’t that different from others after all. Below I listed a few ways that you could implement so that you could stay competitive in the job market.
If you are an Android developer, you could start choosing a job involving Kotlin or Flutter, which are backed by Google. If you are an iOS developer, try to look for a job to get involved with using Swift.
Although older mobile coding languages, such as Java and Objective-C, still have their upsides, the main reason why Kotlin and Swift are created at first is to address the older languages issue. It means that Kotlin and swift provide many safety mechanisms available out-of-the-box while being more concise and expressive than Java and Objective-C at the same time.
 
This point is especially crucial. Each industry has its specific skills to learn and refine. For example, if you are building a stock, foreign exchange, future, or options market app, you will have to understand the WebSocket protocol, which lets you transfer as much data as you like without incurring the overhead associated with traditional HTTP requests.
For the streaming media industry, possessing experience with live streaming protocols such as HLS, RTMP, WebRTC will be a must to deal with streaming-related apps.
These industry skills can really make you stand out, and adds another layer of expertise to your already pretty impressive mobile engineer title, making you more valuable in the job market.
The future of mobile app development will be shaped by how businesses harness mobile technology to solve people’s everyday problems.
Don’t try to fight the irresistible trend, see how you can surf the trend! Exposing yourself or trying to get a job in the field mentioned below will absolutely give you more opportunities as a mobile developer.
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

