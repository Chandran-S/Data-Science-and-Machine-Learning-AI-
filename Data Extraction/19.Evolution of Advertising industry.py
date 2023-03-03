
import requests
from bs4 import BeautifulSoup

# Example URL
url = "https://insights.blackcoffer.com/evolution-of-advertising-industry/"
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
text = ('Evolution of Advertising Industry'
'Advertising can be described as a type of specific content broadcasting to a larger audience; the form can take several different forms, and the intended message can differ from genre to genre. The target for each medium could be different. Advertised content could be in print, radio, TV, or digital formats.'
'We’ll look at how the advertising market has changed over the last ten decade.'
'Advertising is a form of communication that aims to persuade a target audience. Typical advertising messages endorse programs, goods, concepts, people, and companies.'
'First, conventional types of advertising were used to carry out advertising. Let’s take a look at the traditional forms of advertising.'
'Advertising can include in-flight advertising, street furniture, passenger displays, billboards, skywriting, posters, wall paintings, banners, taxi cabs, passenger screens, television, and newspaper advertisements.'
'Other types of advertising include press advertisements in magazines and newspapers. Advertising in the classified section of a newspaper is an example of press advertising. A billboard or digital screen placed on a moving vehicle is often referred to as a mobile billboard.'
'When a brand or product is used in a large entertainment venue, it is known as convert ads or guerrilla advertising.'
'When a soft drink, a watch, or a pair of sneakers is seen or mentioned in a common film, this is an example of this.'
'Ad in supermarket videos, aisles, and on the inside of shopping carts is referred to as in-store advertising.'
'Consumers are influenced by celebrity advertisements because of the power of wealth, fame, and popularity. However, if a celebrity falls out of favor, the use of that celebrity may be detrimental to a company.'
'Religious organizations,political parties, political candidates, and special interest groups are examples of noncommercial ads.'
'These were the conventional forms of advertisement, but as the internet and technology progressed, the advertising industry began to play a role in helping brands establish a digital presence and advertising their products in a new way.'
'The advertising industry is a multibillion-dollar global company that connects producers with customers. According to the research firm eMarkerter, global media advertising spending totaled nearly $629 billion in 2018, with digital advertising accounting for nearly 44% of that amount.'
'For more than a decade, consumers’ perspectives have been shifted in favor of commercials. Advertisements are created based on the preferences of the target audience, and as the population has become more tech-savvy, advertising agencies have shifted their focus from conventional to digital advertising. The internet, as well as the devices, used to access it.'
'Internet advertising has evolved from a risky gamble to the main marketing medium for most businesses. Digital advertising continues to expand by double digits on an annual revenue basis in the United States, with overall spending exceeding $129 billion in 2019.'
'Mobile advertising is a form of advertising that uses wireless devices such as smartphones, tablets, and personal digital assistants to view advertisements. In the consumer goods and retail industries, it is extremely necessary.'
'Mobile advertising contents tailored to particular age groups present an opportunity for the mobile advertising industry. The challenges that the mobile advertising industry faces pose a significant risk of new entrants.'
'Content marketing is an old trend that has resurfaced. Many marketers have struggled to determine how powerful banners and display advertising on other people’s content are.'
'Companies are embedding their marketing pitch within the content itself, rather than serving an ad. This can take the form of publisher-tailored content that the advertiser can support or content that the advertiser publishes directly.'
'There are different kinds of businesses and websites that have used content marketing to grow and flourish in the industry. Content marketing is a trend that has contributed a large amount of income to the advertisement industry.'
'To summarise, the advertising industry has evolved through time and will continue to do so as technology advances, allowing advertisers to reach a wider audience and gain a greater understanding of the people to whom they are delivering material.'
'The advertising industry will continue to develop in tandem with innovation. People are also becoming more jaded when it comes to advertisements, pushing businesses to come up with new ways to convey their messages. However, advertisement has a promising future.')

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

