
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/can-robots-tackle-late-life-loneliness/'


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
text = ('''Can robots tackle late-life loneliness?
In this era, everyone is busy in his life. No one spares time for someone. so robots tackle life loneliness? We have no time to stand and stare. This causes loneliness. Loneliness is due to fast-moving life and hurry and worry. In the 21st century, everyone is spending most of their time earning money. They are day-by-day transforming into money-earning machines. Due to his perception, we are losing our relations as we can’t give enough time to our families. They are busy earning their livelihood. They never get satisfies and always want more. Everyone is in a race of earning and then sends his children far away from himself to make them well settled in order to maintain their status. But one faces the consequences of this kind of lifestyle in his old age. People usually don’t spend time with their children when their children are in their childhood but they yearn for the company of their children in their old age. They feel lonely but have none to talk with. They are still attached to people, places, belongings, and memorable events from the past, although they understand that life cannot continue the same way as earlier, and this may easily result in feelings of loneliness and social isolation. As they become older, they need to be attended to in order to meet their daily needs. Robots can indeed serve the purpose but they can aid only mechanical care. They are just emotionless, feeling fewer entities. Robots aren’t creative. They can work only according to a pre-programmed system and till now no such software has been created that can transfer human feelings into robots. This is the major loophole of using robots to tackle late-life loneliness. They can undoubtedly serve their masters but their masters can’t share their feelings with them as they are incapable to understand or react to them. They can’t change their activity according to the occasion. They can provide anything except emotional satisfaction and without emotional gratification, loneliness can’t be tackled. A master can be attached to his robot but that attachment is only due to their service. They can’t share their thoughts with them. In this era, we are rendered lonely and in late- life feel the need for a companion with whom we can talk. We feel a need for someone who can listen to us, react, and advise us to tackle our day-to-day life problems. But we can’t find one. We have, undoubtedly, achieved great progress in terms of technology but we can never fill the void in the heart of a person struggling with his late-life loneliness with our advancements. We are in a kind of mental trance in which we experience fame, progress, wealth, etc. but when we gain our consciousness, we find ourselves lonely in this world. Relations in life are actual wealth in a person’s life. We can never enjoy such a bond with a robot. It will follow our commands, take our appropriate care but can’t react to our emotions. It can’t console us. When a man lives in loneliness for a long time, it eats up his conscience and transforms him into a machine. Robots can never become our friends, crack jokes, weep our tears, or establish an emotional connection with us. They can’t understand our feelings. Many people go into depression. Depression or the occurrence of depressive symptoms is a prominent condition amongst older people, with a significant impact on their well-being and quality of life. They remain sick and loneliness directly impacts longevity. Lonely people often think that they are no longer needed in this world and thus they want to die. It impacts their mental, psychological, social, and physical health. Robots prove to be useless in these matters. They can provide motivation, only if they have the software to do so but can’t, themselves, react to such a situation. They don’t know about anger, happiness, or sadness. They don’t themselves bring food when their master is hungry until commanded to do so. They can’t even offer a glass of water themselves. A man living without relations and without fellow feeling no longer remains a human being. He becomes none less than a machine as, due to his loneliness, he becomes mentally ill and goes into a condition like trauma where he no longer enjoys nature’s blessings, becomes happy or sad as he loses the ability to react and hence can’t act. It’s a dangerous situation. We face a plethora of challenges in assisted living facilities such as not being addressed to emotional needs, being neglected, and forcing a withdrawal from social activities. Relations in life, interaction with fellow beings, sharing of joy and happiness, and fellow feeling make a man from mere a ‘being’ to a ‘social being’ and finally a human being.
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

