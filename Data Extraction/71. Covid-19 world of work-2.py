
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/how-will-covid-19-affect-the-world-of-work-2/'



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
As business close to help prevent transmission of COVID-19, financial concerns and job losses are one of the first human impacts of the virus;
COVID-19 is in decline in China. There are now more new cases every day in Europe than there were in China at the epidemic’s peak and Italy has surpassed it as the country with the most deaths from the virus It took 67 days to reach the first 100,000 confirmed cases worldwide, 11 days for this to increase to 200,000and just four to reach 300,000 confirmed cases – a figure now exceeded.
In recent weeks, we have seen the significant economic impact of the coronavirus on financial markets and vulnerable industries such as manufacturing, tourism, hospitality and travel. Travel and tourism account for 10% of the global GDP and 50 million jobs are at risk worldwide.
Global tourism, travel and hospitality companies closing down affects SMEs globally. This, in turn, affects many people, typically the least well-paid and those self-employed or working in informal environments in the gig economy or in part-time work with zero-hours contracts. Some governments have announced economic measures to safeguard jobs, guarantee wages and support the self-employed, but there is a lack of clarity in many countries about how these measures will be implemented and how people will manage a loss of income in the short-term.
Behind these statistics lie the human costs of the pandemic, from the deaths of friends and family to the physical effects of infection and the mental trauma and fear faced by almost everyone. Not knowing how this pandemic will play out affects our economic, physical and mental well-being against a backdrop of a world that, for many, is increasingly anxious, unhappy and lonely.
Fear of the unknown can often lead to feelings of panic, for example when people feel they are being denied life-saving protection or treatment or that they may run out of necessities, which can lead to panic buying.Psychological stress is often related to a sense of a lack of control in the face of uncertainty.
In all cases, lack of information or the wrong information, either provided inadvertently or maliciously, can amplify the effects. There is a huge amount of misleading information circulating online about COVID-19, from fake medical information to speculation about government responses. People are susceptible to social media posts from an apparently trustworthy source, often referred to as an “Uncle with a Masters”-post, possibly amplified and spread by “copypasta” posts, which share information by copying and pasting and make each new post look like an original source, as opposed to posts that are “liked” or “shared” or “retweeted”.
Sadly, criminals and hackers are also exploiting this situation and there has been a significant rise in Coronavirus-themed malicious websites, with more than 16,000 new coronavirus-related domains registered since January 2020. Hackers are selling malware and hacking tools through COVID-19 discount codes on the darknet,many of which are aimed at accessing corporate data from home-workers’ laptops, which may not be as secure as outside an office environment.
Social distancing and lockdowns have also prompted altruistic behaviors, in part because of the sense that “we’re all in this together”. Many people report being bored or concerned about putting on weight; others have discovered a slower pace of life and by not going out and socializing have found more time for family, others, and even their pets.
The downside of self-isolation or social lockdown are symptoms of traumatic stress, confusion and anger, all of which are exacerbated by fear of infection, having limited access to supplies of necessities, inadequate information or the experience of economic loss or stigma. This stress and anxiety can lead to increased alcohol consumption, as well as an increase in domestic and family violence.In Jingzhou, a town near Wuhan in Hubei province, reports of domestic violence during the lockdown in February 2020 were more than triple the number reported in February 2019.
Health measures must be the first priority for governments, business and society. It is important for businesses to show solidarity and work together to protect staff, local communities and customers, as well as keeping supply chains, manufacturing and logistics working.According to research, “my employer” is more trusted than the government or media. Daily updates on a company website with input from scientists and experts are recommended to counter politicized messages in the media and from governments. This is particularly true for large companies that have the capacity to do this.
Messages about what businesses are doing for their employees and in their communities is also important. Some companies are helping schoolchildren from vulnerable families who can no longer get a school meal; others are providing public health messages about effective handwashing. Even CEOs can show they are working from home and self-isolating, while still being effective in their leadership.
Following WHO advice, there is a need for the business community to move from general support to specific actions and focus on countries’ access to critical supplies, including a “Community Package of Critical Items” (a list of 46 items that all countries need). Of these items, 20 are either not available locally or available stocks are too limited. These missing items fall into four categories:
The call for action is for more money, to work with manufacturers to create capacity and to organize purchasing so there is guaranteed access, especially for poorer countries with less resilient public health systems. The concept is to create a global security stockpile of supplies and equipment, an effort that needs:
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

