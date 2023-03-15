
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/lessons-from-the-past-some-key-learnings-relevant-to-the-coronavirus-crisis-4/'


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
text = ('''Lessons from the past: Some key learnings relevant to the coronavirus crisis
As a race, we are not ready for any kind of crisis. We have fire extinguishers in buildings, but barely any knowledge of how to use it. We convince ourselves that we are prepared but in the face of adversity we stand dumbfounded and give the simplest excuse, that of being in ‘unprecedented times’. The same thing has happened with the coronavirus crisis. If we look back, we can see that this crisis was not avoidable. There was a lot we could have done, but because we underestimated it so much, the situation has now blown out of proportion and cost us thousands of lives, millions of jobs and a massive hit to the world economy. We may be in a unique condition but there is still a lot that we can apply from past experiences and previous calamities. 
First, let’s look at this in a reflective mode. What are the things we could have done to avoid this crisis in the first place? Controlling the origin of a virus is difficult but preventing it from spreading is not. One of the biggest contributors to this unforgiving crisis is the lack of communication or even worse, the propagation of fake news. Take India for example. Clear communication about the virus was only initiated after a few cases had been identified. We were aware of the hazards of this virus since before and should have acted sooner. The spread of accurate and timely information is a must. If we had spread awareness earlier, we could have controlled the number of cases better, like New Zealand and Ireland have done. Secondly, in a situation of crisis, the first thing to spread is fake news. It is imperative that the government, social media agencies and people themselves act more responsibly. Since there is no accurate and transparent information, a lot of uncertainty exists in the society. This impacts the confidence of consumers and investors in the market, which in turn leads to an economic slowdown. Past experiences with financial breakdowns and recessions have shown us that retain market trust can only occur with transparency and accuracy of information. Organisations should have acted earlier and communicated more effectively with their employees. 
Moreover, we must also understand that specific sectors take the hardest hit during lockdowns. As we have witnessed in earlier lockdowns, curfews, national emergencies and pandemics, the first segment of society to take a hit is the daily wagers and labourers. Not only do the prices of goods increase but their income also takes a decline in such a crisis. In such situations, they can’t even afford basic amenities. India witnessed a mass exodus recently, with thousands of people trying to return to their villages on foot. This only leads to more chaos and a decline in containment of the disease. The government should have accounted for such factors before announcing a lockdown with a four hour notice. They could have made better arrangements for food and water in the metro cities itself or giving them safer means to travel home. This is something that they can still include in their relief programs.
Government intervention is a must right now. This can be done in two ways. First, is to create more employment opportunities. Governments used this strategy in the SARS and Ebola Virus emergency. Health, water, sanitation and hygiene are the most services in the times of health emergencies. Intense investment for these services and infrastructure can provide immediate jobs. Secondly, if governments use fiscal and monetary policy to fuel demand and maintain current living standards, it can decrease the propensity of a huge fall in economic growth. A fiscal stimulus by the government can make all the difference. Having better avenues for cash transfers, easier loans and more exchange will automatically help. A fall in the growth rate is unavoidable as lots of sectors have no choice but to pause all work. However, industries that can provide high returns even now must be encouraged. For example, IT sector, health, food and agriculture, etc.
Moreover, governments and businesses should use the most important tool in their hand, the internet! Even in the times of physical social distancing, we are still connected to people across the world and are socialising at our normal rates. Governments should use these mediums to maintain public spirit, communicate policy strategies and inform people. Businesses should use these means to continue working and maintain employee morale. To have balanced crisis management and to prevent panic in the masses, social dialogue and engagement at every level is imperative. Use these mediums to educate people about the importance of social distancing, hygiene and steps to be followed.
All diseases are different and the scientific approach to tackle them varies. However, with most epidemics and pandemics some common steps can be followed. Our past experience with fast spreading diseases has shown that the easiest and most effective way to curb it is social distancing. Countries have taken impressive steps to ensure social distancing and must keep this up for even a few weeks after curbing the disease. Maintaining sanitary habits and hygiene can protect us from infections and increase our immunity. While governments and medical agencies look for ways to tackle the virus at a macro level, we must continue to follow all measures on an individual level too.
We must also pay a lot of attention to mental and physical well-being now. In the past, when economies took a hit due to the 2008 recession or calamities, the first thing to break was people’s self-esteem. Suicide rates, amount of domestic abuse and cases of depression increased manifold. We have to take effort and ensure that we try and live as much of a balanced life as possible. We should stay connected to our friends and not be ashamed of reaching out to others for help. Social distancing is not easy and can take a toll on many. Lots of special helplines have been set up for the same and people must use them if needed. Governments have also given code words for victims of domestic abuse to report crimes in countries like France and Spain. We must remember to stay mentally healthy and calm during a crisis as that is the most important thing.
I don’t believe that every crisis is the same. Of course, there are a lot of unique things about the coronavirus pandemic that we had not expected to ever see. Even though these are different times, there are still lots of things we can apply from our past experiences. We cannot find a cure for the virus but we can find ways to maintain the economy, our mental peace, and prevent the spread of the virus. The past has a lot to offer if we ponder. We must learn to understand the mistakes of the past and better them in the future. Our past experiences with the different crises have taught us that fiscal stimuli work, maintaining employee morale is important and that social. Engagement is imperative. If we use these tools efficiently, we can ensure that people only have to fight the virus and not worry about the multitude of repercussions that come with it.
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

