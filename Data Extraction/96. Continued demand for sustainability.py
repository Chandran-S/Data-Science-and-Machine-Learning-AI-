
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/continued-demand-for-sustainability/'



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
text = ('''Continued Demand for Sustainability
The business of business is no longer to do just business
or increase the bottom line to maximize shareholder value. Rather, the concept
of business is moving towards a new dimension of sustainable business, the
triple bottom line. People, planet, and profits are the core
ideologies that are rooted in sustainable business. Sustainability is taken
into account when companies want to create long-term value creation along with
strategies that promote the longevity of the company. As corporate
accountability rises, expectations and need for transparency among stakeholders
increases therefore companies have started to recognize the need to be
sustainable to stay alert and alive. 
Business
globalization that has happened over the previous few decades has made some
companies more powerful than some national governments, making it easy for them
to exploit inexpensive labor, plunder natural resources, causing severe impacts
through pollution on the natural environment, human health, and biodiversity.
Unfortunately for them, their horrible past has been catching up with them like
in the case of child labor issues of IKEA & Nike, Rana Plaza accident in
Bangladesh affecting Zara, H&M and other clothing brands, environment
pollution by BP, Shell, Exxon Mobil etc. With the emergence of internet and
social media these practices can no longer be covered up and silenced, the
world has become more educated and less tolerant and therefore with every
misdemeanor that is committed brand equity takes a hit. The media is fast
picking up on cover-ups, half-truths, and bad corporate behavior and demanding
accountability and transparency from corporates. Therefore to survive, companies
are compelled to adopt sustainability and bring forth the rules for their
suppliers as well. 
Many
of the irresponsible company practices and disasters that are witnessed in that
last couple of decades were motivated solely by short-termism- the desire
for instant gratification- and the appeal of short-term performance
incentives. For instance, deferred maintenance and slack leadership were to
blame for the 1984 leaks from the pesticide plant Union Carbide India Ltd in
Bhopal. Estimates placed more than half a million casualties from the gas
discharge of the plant. Finally, Union Carbide Corporation’s 1989 litigation
payout came out to today’s equivalent of nearly one billion US dollars. Another
example where the company was forced to shut down operations was Coca Cola at
its Plachimada plant in Kerala. The local communities faced acute water
shortage after the commissioning of the plant and the company authorities
blatantly ignored the community woes. These businesses stated above didn’t
respect stakeholder engagement which ultimately lead to their untimely demise.
Today’s
issues cannot be solved without participation from all stakeholders, and
companies with influence should be the torchbearers of change. They need to
engage with regulators, communities, societies, suppliers, and NGOs for
effective and desired outcomes. Issues where engagement are required include
population growth; global middle-class growth; decline in ecosystems; water
scarcity; food safety; material resource security; higher global energy demand;
changes in geographic patterns of energy consumption; and increasing climate
change regulatory interventions.
By actively pursuing the triple
bottom line, the essential possibilities available are:
In order for companies to commit
to sustainability they should have the below objectives in place:
Sustainability -triple bottom line- can drive a company’s achievement beyond shareholder value creation by building corporate shared value at its core and help in addressing social &environmental problems. Several investors today use ESG metrics to evaluate the ethical effect and sustainability practices of an organization. Investors are looking at variables like the carbon footprint of a company, water use, community development efforts, and diversity board before investing. Companies have started responding to investors by publishing their annual sustainability reports. Research indicates that businesses with elevated ESG scores have reduced debt and equity costs and that sustainability projects can contribute to improving economic efficiency while encouraging government assistance. It’s only a matter of time, the flow of sustainability nourishes businesses in achieving holistic development for the environment, people and its profits.
Blackcoffer Insights 12 | Sanjana Jose Varghese | IIM Lucknow 
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

