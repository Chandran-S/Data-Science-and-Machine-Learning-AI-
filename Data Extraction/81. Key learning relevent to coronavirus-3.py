
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/lessons-from-the-past-some-key-learnings-relevant-to-the-coronavirus-crisis-3/'


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
The current topic that is burning the world alive is the deadly coronavirus. Let us first understand what this coronavirus actually is. According to the World Health Organization (WHO), the Coronavirus disease which is called the COVID-19 is an infectious disease which is caused by the newly discovered Coronavirus. The COVID-19 virus is a virus that affects different people of the world in different ways. The COVID-19 disease is a respiratory disease and most of the infected people will develop mild to moderate symptoms and recover without even requiring special treatment.  The people who already have underlying medical conditions and those people who are over 60 years old have a higher risk of developing severe disease and death.
Common symptoms include:
Other symptoms include:
The people living in this world who have only mild symptoms of this deadly disease but are otherwise healthy should self-isolate and contact their medical provider or a COVID-19 information line for advice on testing and referral. The people who have a fever, cough, or difficulty breathing should call their doctor and seek medical attention.
Some key lessons from the deadly disease Coronavirus: In the last 3 months, there has been a drastic change to human life all around the world. The coronavirus outbreak happening in the world is no more just an epidemic but it is also considered as a global pandemic now. The people everywhere in the world are being asked to stay home and stay away from other people so as to reduce the risk of infection.
The deadly disease Coronavirus has brought with it a wave of negative outcomes, terrible illness, and death, but at the same time it has also highlighted some important life lessons. Some of them are as follows:
1. You should be willing to trade some of your freedom for the greater good of the public: There is no doubt for anyone that it has been difficult for everyone in the world to stay at home. Many people complaining about feeling bored and aimless. Some people may even have the feeling that it is a breach of their individual rights, being made to stay home. However, when it comes to the greater good, one should always be willing to sacrifice a little bit of that freedom. A balance between individual rights and public safety is an ever-changing thing. Trade a little bit of your freedom for the greater good of the public.
2. You should wash your hands, whether there’s a virus or not: General hygiene is always important. Not just when there is a virus. You should know the drill by now. Wet your hands. Lather them with soap. Scrub for 20 seconds. Rinse off. Dry with a clean towel. Washing hands on a regular basis are really the best way to keep safe because soap is a very effective way to kill viruses.
3. Working from home should be an option for many: In this time of the global outbreak of the Coronavirus disease, many people have learned that their jobs were possible to do from home. Most jobs have a certain amount of work that can be done remotely. Without the virus in place, there should still be some system in place that will promote work-life balance.
4. Taking that sick day from the work could save lives: If you are feeling sick at any day in this situation, it is highly beneficial that you just stay home. It is a common feeling among lots of people that their office environment does not encourage taking sick days. Many people want to appear like martyrs to their managers. This mentality needs to stop. If you are sick, just stay home.
5. The Internet should be a basic right: According to a study done by the University of Birmingham, the right to Internet access, also known as the right to broadband, should be considered a human right. Many people from all over the world are unable to get online, particularly the people who live in developing countries, and so they lack meaningful ways to influence the global players shaping their everyday lives. In addition to this, during times like these, it is especially important to be able to contact family, friends, and work from home if necessary. The Internet is the only way to do so.
6. The doctors and the researchers need to be paid better: If this scary time has taught us anything, it’s that doctors and researchers will be the ones who get us out of this mess. The doctors and the researchers are the only ones who are working day and night to drive the recovery of the world. At the moment, hundreds of scientists scramble to find a coronavirus treatment. It is high time for us that we need to re-evaluate how much money Hollywood actors, pro-athletes, and politicians make and instead pay the scientist and the doctors the salary they deserve.
7. Everyone should know how to cook: The situation that has forced everyone to stay at home has forced many people to learn, re-learn, or re-ignite their love for cooking. Learning how to cook is to have one of the most important skills in life. You depend on yourself. It teaches you self-sustainability and you save a lot of money. These days, hundreds of people sharing social media posts of their delicious meals. They are re-discovering the wonders of eating in.
8. The importance of talking to friends every day: Now that we can’t go out and keep busy, the best way to combat loneliness is to be in regular contact with friends and family, by chatting over the phone or video chatting. This is the time to have long talks and deep conversations. Don’t forget human connection during these crucial times. Call your grandma!
9. Learn to appreciate nature: If you live near a spacious outdoor area, like the desert or an empty road lined with trees and you realize it is the only safe, surface-less space to take a walk in, then you start to realize how beautiful nature actually is. The point is not to remain indoors, but to avoid being in close contact with others. When you do leave your home, whether it is for a walk in the desert or a run on your street, make sure to wipe down any surfaces with which you come into contact and at the same time also avoid touching your face and frequently wash your hands.
10. Learn how to be content alone: It is so hard for some people to just be still and do nothing. Being alone, especially for extroverts can be exhausting and lonely. Social distancing can be very difficult, but it can also teach you a lot about yourself. You learn how to keep yourself busy. Your body and mind is your home and you have to learn how to love it and live with it.
Conclusion: It is rightly said that it is important to learn from the past. The deadly Coronavirus disease that is taking the lives of so many people across the world, has taught so many life lessons to the people living in it. These lessons are not meant to be forgotten but are to be remembered by human beings in order to make the Earth a beautiful place to live in. Lastly, I would like to conclude by mentioning that “Stay at Home. Stay Safe. Save Lives”.
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

