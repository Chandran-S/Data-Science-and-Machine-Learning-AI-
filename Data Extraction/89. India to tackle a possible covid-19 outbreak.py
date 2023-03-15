
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/how-prepared-is-india-to-tackle-a-possible-covid-19-outbreak/'


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
text = ('''How prepared is India to tackle a possible COVID-19 outbreak?
Frankly, no! The state of the public healthcare system, still vast, is very fragmented and cannot scale up rapidly enough to face the challenge of a pandemic such as COVID-19. The origin of public healthcare lies in the British colonial period where such facilities were provided only to British officers and their families stationed in India. From the 1960s onwards, the independent government of India stressed more upon healthcare and it led to the creation of the modern public healthcare system in India. The healthcare system is organized into primary, secondary, and tertiary levels. At the primary level are Sub Centre’s and Primary Health Centre’s (PHCs). At the secondary level there are Community Health Centre’s (CHCs) and smaller Sub-District hospitals. Finally, the top level of public care provided by the government is the tertiary level, which consists of Medical Colleges and District/General Hospitals. The number of PHCs, CHCs, Sub Centre’s, and District Hospitals has increased in the past six years, although not all of them are up to the standards set by Indian Public Health Standards. But there are still critical challenges to the existing system.
One of the major areas is the lack of awareness of diseases, especially communicable diseases among the general population of India. This challenge becomes more complicated at the lower strata of society, especially poor households, migrant laborers and homeless people who do not have the proper channels of communication available to get the necessary information from the health experts. Add low educational status, poor functional literacy, low accent on education within the healthcare system, and low priority for health in the population, among others and you see the magnitude of the problem. According to a recent study initiated by the faculty and students of IIT-Hyderabad and IIT-Bombay, in Tier-1 cities, it was found that about 12 percent of the respondents switched from public to private mode during the third week of COVID-19. This modal shift was about 9 percent in Tier-2 cities and about 7 percent in Tier-3 cities. Moreover, nearly 48 percent of people said that they did not travel to work during the third week of March, whereas 28 percent had the same frequency of travel to work. When inquired about the cancellation of trips between the cities using major modes of transportation, around 18 percent said they cancelled their flights whereas, 20 per cent of respondents cancelled their train journeys. This indicates that awareness about COVID-19 is higher in Tier-1 cities, in comparison with Tier-2 and Tier-3 cities. But national level, state level and door-to-door campaigns along with targeted media campaigns may enable the knowledge of the pandemic to spread fast among the local populace.
The second challenge is access to public healthcare in India. Access (to healthcare) is defined by the Oxford dictionary as “The right or opportunity to use or benefit from (healthcare)” Physical reach is one of the basic determinants of access, defined as “ the ability to enter a healthcare facility within 5 km from the place of residence or work”. Using this definition, a study in India in 2012 found that in rural areas, only 37% of people were able to access IP facilities within a 5 km distance, and 68% were able to access out-patient facilities. Furthermore, it was postulated that in general, the more rustic (rural) one’s existence – the further one lives from towns – the greater are the odds of disease, malnourishment, weakness, and premature death.
The third is the planning portion related to the preparedness of the pandemic. Planning for a pandemic such as COVID-19 has never been done extensively in India as we had no recent history of such a communicable disease which spreads rapidly in crowded locations, which about a quarter of the population live in. The recent national lockdown called led to an increase in police violence who have not been sensitized how to deal in such a crisis and the humanitarian crisis facing migrant laborers, an important part of the Indian economy, who have lost their jobs and facing a lack of public transport, have to force to walk thousands of miles to reach their native villages. The ongoing slump in the economy has been further exacerbated by the pandemic and the resultant lockdown. And there have been growing complaints by healthcare professionals, who are fighting in the front of this pandemic, of lack of Protective Personal Equipment’s (PPE’s) to protect themselves and their patients from COVID-19. The lockdown has hit HIV+ and chronic patients hard who need constant medication to maintain their fragile condition. Elderly people, who are often alone and have medical conditions, have been left without essential support. Ironically, these are the same group of people who are at an immediate threat from COVID-19.
I am optimistically hoping, based on the ingenious and creativity of our citizens, that we tide over these tough times and fundamental changes to our healthcare system are made with this sober learning experience to prevent a repeat of what is happening today.
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

