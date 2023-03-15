
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/impact-of-covid-19-coronavirus-on-the-indian-economy/'

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
text = ('''COVID-19 Vs Indian Economy
India is on the verge of an unprecedented economic as well as humanitarian disaster from COVID-19. India is facing economic pressure due to the coronavirus outbreak. This outbreak has resulted in a steep slowdown in the Indian economy, whereby India’s GDP has reduced by 4.5% as seen by current statistics. With COVID-19 being into the picture the Indian economy is going through a severe slowdown and on top of that multiple such problems hitting the world of work from many such directions. Companies and industries are finding it difficult to sustain in such an environment of the financial crisis. Industries are being urged to take up certain tough decisions like layoff, retrenchment, compulsive leave without pay and many alternative cost-effective methods to sustain the present economic crisis like cutting down the salaries, handing over the pink slip to employees and opting best possible. The sudden outbreak has presented new roadblocks for the Indian workforce and especially for the daily wage and contractual workers.
According to the latest economy forecast by the UN, the probability of countries entering into the recession and companies going bankrupt has increased and India is not likely to “remain decoupled” from the global meltdown. Lockdown across the country is causing significant disruption across multiple sectors including manufacturing, tourism, aviation, real estate, etc are the worst affected. Temporary lockdown resulting in the closing of shopping malls, has affected the Indian economy in a  disastrous manner, adversely as a result of which consumption of essential items has reduced which in turn has given a terrible blow to the retail sector.   
 According to the statistics from the Economic Times of India, live event industries have seen an estimated loss of ₹3000 crores. A number of young startups have been affected as the funding has fallen and there would be a lack of investors in the market. The disruption of the economy is much starker and alarming than the global financial crisis of 2008, which hit the Indian financial sector. Besides that back in 2008 the Indian economy was much better placed to handle the crisis. According to the data in 2019 the GST collection was expected to be approximately 7.4 lakh crore while the government was only able to collect a total of 5.8 lakh crore that is a total loss of 1.6 lakh crore, the government also faced a setback in the income tax sector as the expected revenue to be collected was estimated to 5.2 lakh crore but the government faced a shortfall of 50,000 crores. The above discrepancies have resulted in insufficient funds with the government Clearly the government does not have sufficient funds to provide relief to the people mainly the labor class, contractual workers and the people working on the basis of daily wages. 
Other major industries whose figures of losses are alarming are the Hospitality and tourism industry which employs approx. 4 crore people. In the next 10-12 months it expects 12 lakh job losses which may act as a major factor in revenue lo and revenue losses of ₹11,000 crores. The aviation industry is worth ₹2.2 lakh crore, employing 3.5 lakh people, they expect a revenue loss of ₹4,200 crores just between April and June. India’s retail industry is the total worth of ₹59 lakh crore employing around 4.6 crore people, if the pandemic crisis lasts 3 months further it expects 1.1 crore job losses. The restaurant industry employs 73 people, of whom 14%-15% may probably lose their job. The real estate industry is looking at an approximate 35%-40% job loss. Ride-hailing industries meaning Ola and Uber have approximately 5 million driver-partners, the crisis has led to a drop of 40%-50% in the business. These figures would get even more disturbing in the next few weeks as the country heads into total lockdown. Top car manufacturers like Honda, Hyundai have shut down their car productions. Many other such companies have shut down their production until further notice. This would again lead to more lay-offs. The stock market in India has also seen a breakdown since the lockdown   
Quarantine and lockdowns have disrupted the chain supply across the whole world. As the trade between the countries has come to a halt, it has affected the global economy as well as the country’s GDP. While some industries in India depend on other countries for their raw material, for example the electronics market, it depends on china majorly for its raw material, has seen a major setback as trade between the two countries trade has been seized.  
The outbreak of the virus has placed tremendous responsibility in the hands of the government. The Reserve Bank of India (RBI) along with the government intends to implement below-mentioned measures to deal with the looming economic crisis such as: 
An Economic response task force was
announced on 19 March 2020, led by the finance minister of India Nirmala Sitharaman,
to tackle the financial crisis. As of announced on March 26, a $ 23   billion dollar package for fighting the
economic pandemic has been sanctioned. The spending proposed in the package
would amount to about 0.5% of the estimated GDP, while the other countries have
a package of about 4%-5% of its GDP.
Everywhere in the world, governments are recognizing that this is no time to worry about fiscal deficits. Instead, they have to do “whatever it takes’’ to come out of this extraordinary crisis.  Immediate and necessary actions are indispensable not only from the Indian government but also from every individual to prevent this health pandemic from turning furthermore into an even greater economic disaster. 
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

