
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/coronavirus-impact-on-the-hospitality-industry-5/'

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
text = ('''COVID-19 Impact on Hospitality Industry
In December 2019, a novel coronavirus strain (SARS-CoV-2) emerged in the city of Wuhan, China. The disease called Covid-19 spread to almost every nation in the world creating widespread havoc and disruption in routine life. WHO recognized it as a pandemic on March 10, 2020. As of April 19, 2020, worldwide around 24 lakh people have been infected and more than 1.6 lakh people have succumbed to Covid-19.
Though a vaccine hasn’t been developed yet, the spread of coronavirus can be stopped by washing hands frequently, covering one’s mouth while coughing, and practicing social distancing from other people. Countries across the world are using lockdowns as an effective way to stop the spread.
 29 kb of RNA has brought the world down to its knees. One-third of the global population has been in lockdown. Schools, colleges, businesses, services that are considered non-essential have been put on hold for an indefinite period. Even after lifting restrictions, a global recession even worse than the one in 2008 is going to be expected.
Now arriving at the main field of discussion, i.e. Hospitality Industry. How COVID-19 Impact Hospitality?
Propulsion was building for 2020 to be a year of focal collective action on sustainability. The Covid-19 outbreak has emphasized more than ever, the importance of future-proofing business for growth and resilience.
The hospitality industry includes various services like lodging, food and beverage, events, tourism, transportation, theme parks, etc. Before COVID – 19 prevailed, we all know that this industry was in much demand and was also one of the major sources of the rising economy. During vacation and festive periods, these were always full of people worldwide. As of now, due to the havoc created and ongoing lockdown they are bearing many losses and is one of the main reason for the decreasing economy.
Have you ever thought of hotels been converted into hospitals, quarantine centers, and isolation centers?  Yes, my dear people! Amidst this outbreak, many of the hospitality industry properties, be it tiny or enormous have come forward to help the nation fight this life-menacing virus.
Taj hotel in Mumbai is attentively aware of its responsibility towards the community and has opened its doors for the major frontline workers, i.e. medical fraternities to stay at their place while they combat the spread of this treacherous virus.
Also, many of the government bodies have been transformed to provide shelter to the homeless. Recently, in Vadodara – Gujarat the building which was built for the employees aiming to work for the major Mumbai-Ahmedabad bullet train project has been come forward to make it an isolation center for doctors and nurses.
“Railways – the lifeline of citizens.” As railways have started their service years ago, from that period they are being considered as the lifeline of citizens in many metropolitan cities. Today at the time of the global pandemic, railways haven’t backed off from the position of the lifeline. Yes, my dear friends, you have heard it right! Railway coaches in India have been turning into the hospital and isolation centers, marking their presence by helping the government to increase the number of isolation beds. Also, many special parcel express trains are running in the entire nation to fulfill the needs of people by transporting essential commodities and goods.
Hospitality has been one of the most innovative industries in the crisis so far. Connections have been rapidly formed to donate food and beverages to local charities from various hotels. Many of the hospitality bodies are standing in solidarity with the communities affected by this threatening disease by lighting their windows in the shape of a heart, seen in hotels across the world.
With this outbreak, there is a sharp drop in tourists worldwide as the aviation, railways and public transport buses have come to a standstill due to severe government restrictions.
Today it is clear that hospitality industries must be prepared for various situations like a pandemic, climate change, etc. as of now they are facing devastating and disastrous Covid-19 impact on its various sectors.
The debt being the normal capital intensive component has to be serviced by payment of interest on debt and repayment of debt. Hotels being labor-intensive, have lots of fixed costs such as wage bills, besides paying government levies, minimum load charges, etc. The earlier Indian hospitality industry was on average witnessing 65 to 70% of occupancy till the end of February. The first few days of March were fine, once things started accelerating, the occupancy has gone down to a severe minimum. Also as soon as the pandemic ends by God’s grace, the hospitality industry will still face some loss as the tourists will be much lesser in the beginning months.
As soon as the crisis gets over, the hospitality industry will have a very crucial role to play in rehabilitating lives within their local communities. Millions of people will be unemployed leaving them at high risk of poverty and exploitation. As the industry starts to recover, hospitality will be one of the sources to increase employment to the needy ones taking the poverty level to a minimum.
Within India, after the crisis over, people should travel to the less known and highly economically affected destinations to help the hospitality industry overcome the loss they bore due to Covid-19.
Hence, the hospitality industry suffered and is suffering a lot due to this pandemic and despite that, it is also helping the nation to win against this dangerous disease. thus COVID-19 Impact on Hospitality 
“Let’s be grateful to all the frontline workers saving the entire nation and also be thankful to all the various sectors of the hospitality industry who have come forward to help and serve the nation.”
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

