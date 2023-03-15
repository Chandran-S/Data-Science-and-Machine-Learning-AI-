
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/coronavirus-impact-on-the-hospitality-industry-2/'
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
text = ('''How Coronavirus Impact on the Hospitality Industry
While Coronavirus has affected the life of every individual, it has also affected various industries for better or for worse. While certain industries like Pharma, Mobile and Web-streaming industries, Fitness and Teleconference apps have boomed during this pandemic, industries like Tourism, Fast-Food Industry, Retail and Leisure sectors have suffered the worst hit due to Coronavirus Impact
During this lockdown to prevent the spreading of Coronavirus Impact, almost all of the companies have adopted a work-from-home culture and have used various Teleconference apps to hold meetings. Teleconference apps are also being used by college institutions and universities so that the studies don’t get disrupted. Also, confined to their homes, people have started engaging more in fitness apps, and web streaming services like Netflix, YouTube, and Amazon Prime are at their zenith.
While these industries are thriving in this pandemic, the most affected industry during this time is the Hospitality Industry.
Hospitality Industry embodies various sectors like transportation, traveling, airlines, hotels, and lodges, etc. Hotels and traveling are also a major part of the Tourism Industry. The tourism and Hospitality industries may appear somewhat similar, but they have their subtle distinctions. Tourism mainly involves traveling and services to those who are away from home. The hospitality industry involves businesses and services that are concerned with leisure and enjoyment and mainly run on customer experience and satisfaction. Their customers may also be local inhabitants along with tourists.
The hospitality industry can be majorly divided into Accommodations, Travelling, and Food and Drinks. 
Accommodations include hotels, resorts, hostels, motels, etc. The best of the hotels and resorts that were basking at their peeks, now have no option but to lay off their employees. Many hotels are down to zero occupancies and have led to an immense loss for the hotel establishments. The same is the case for resorts, hostels, motels, etc. that provide accommodations. Since there are no occupants, there is no money inflow, which is leading to the layoffs of employees as the owners are not able to provide paychecks anymore.
Businesses in the accommodation sector that have just started, are finding it difficult to cope up with this situation. Various governments have extended small business bridge loans to cope up with this situation. The accommodations sector has no other way to cope with this situation but to wait for the situation to improve so that they can improve their business.
Traveling restrictions are imposed by various governments to avoid the spreading of Coronavirus. Travel agencies, car rentals, tour operators, flights, and airlines have been hit quite severely by COVID-19. Due to the fear of spreading Coronavirus, people have limited their public outings or have stopped stepping out altogether unless to shop for daily essentials. Coronavirus Impact on travel that ban lays out restrictions on who can travel from where to where. Many international and domestic flights have been suspended to tackle the Coronavirus crisis. Havoc is created across the aviation industry.
The months from March to May are said to be the busiest times of the year for the tours and travel industry. But people are reluctant to travel in the fear of catching Coronavirus or getting stuck abroad if the governments announce restrictions, thereby overstaying their visas that can imply their future travels abroad. Airlines and tourism agencies have stopped marketing and advertising as it would be irresponsible and does not make sense during this crisis.
To cope up with this situation, airlines like Virgin Atlantic and American Airlines have started deploying passenger flights to carry cargo, while various airlines have laid off. Commercial travel has dropped tremendously. Also, there is a recent drop in jet fuel prices. It is being estimated that many airlines are going to go bankrupt if this situation persists.
Apart from airlines, the cruise industry and travel agencies also take a toll this year. They are facing cancellations of bookings, and some new travel agencies are at a risk of closing down rendering all their employees either losing their jobs or going on unpaid leaves in this time of crisis.
It is predicted that once the pandemic is over or the situation is under control, the travel and tour industry can take about 10 months to return to normalcy. However, this outbreak may affect the entire tourism industry’s future.
Many foods and drinks services fall under the hospitality industry as they offer people services that provide them leisure time. Their customers not only include tourists but locals, passerby’s, etc. These mainly include restaurants, tea and coffee shops, bars and pubs, catering, etc.
Catering is provided at places mainly where there is a social gathering, to deliver food and drinks services where it is not provided or not according to the organizer’s preference. Due to this pandemic, many countries are in lockdown, and even those that are not in lockdown are not allowed to hold public gatherings as a safety precaution. Due to the avoidance of public gatherings and events to maintain social distancing, catering services are halted for now.
Restaurants, tea and coffee shops, pubs, and bars have all closed down not only for the safety of their customers but also keeping in mind the safety of their employees and workers. However, a few restaurants that offer food delivery are still working, by taking all the necessary health and sanitation precautions to prevent the spreading of Coronavirus. The number of such restaurants seems to be declining with the increasing cases of COVID-19 as the customers are becoming more and more afraid and cautious about the situation. Though, quick-service food deliveries can cope better than the full-service restaurants during this crisis.
Because of this unprecedented crisis, all these industries that are being affected are hoping for government support to make it through such tough times. Without some financial support or help from the government or other institutions, it will be difficult for various establishments to survive through this crisis.
It is being indicated that the world will face the worst recession since the Great Depression during this pandemic with so many people losing their jobs, especially those working in the hospitality industry. Thus Coronavirus Impact on different industries
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

