
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/what-is-the-repercussion-of-the-environment-due-to-the-covid-19-pandemic-situation-2/'


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
text = ('''Due to the COVID-19 the repercussion of the environment
Epidemics, in general, have both direct and indirect costs associated with direct and indirect measures adopted to counter (control) the epidemic which generally have both short-run and long-run economic and social consequences. The 2019 Coronavirus (COVID-19) outbreak globally lead a significant setback to the entire globe. The first case was detected on 26th December 2019 in China.  The World Health Organization (WHO) prepared the first diagnostic kit on 14th January 2020.
As the global coronavirus or Covid-19 pandemic continues to take hold every economy is feeling it’s effect with a high degree of uncertainty taking a toll in all the manufacturing sectors. Supply chain management is an important area with many opportunities for our community to contribute in various forms.  According to the data produced by Supply Management’s Report on Business show that PMI (Purchasing Manager’s Index) has declined 1% in the month of February. Mainly there has been a huge contraction in the Petroleum, Transportation and Textile Industries. Although there has been a fall in the inventory but is expected to grow due to supply chain disruptions leading to inefficiencies in material conversion and continued advanced stocking to protect production schedules. A analysis by trading platform Forex claimed that heavily 75% of all companies had already reported supply chain disruptions with more than 80% believing that at some point they would experience impacts as a result of Covid-19 disruptions. The effective shutdown of industrial activity i.e. China, ground zero for the virus presented particularly difficult problems for manufacturing firms worldwide. Imported goods invested by investors set a significant setback.
Several experiments were carried out by different governments across the globe to restrict the spread of this pandemic. Boris Johnson introduced the herd immunity plan in the United Kingdom which was a failure followed by a mitigate model which India also tried initially to stop the spread of the Covid-19 virus. The last and the ultimate model that most of the countries are applying is the Hammer and the Dance Model i.e. the Lockdown Model which is estimated to bring down the caseload by 25-30%. The Lockdown model basically means buying time to prepare the vaccine to kill the virus.
However, the lockdown has certain critical economic consequences which add to the burden of global distress. This can be explained using the standard macroeconomic Keynesian model. The lockdown and the spread of the disease have a direct negative shock on aggregate consumption levels and exports. This leads to a contraction in aggregate demand which leads to a fall in the market rate of interest and aggregate equilibrium output. The secondary effect on the commodity market is via investment demand function which can either increase owing to a fall in the market rate of interest, or fall owing to the contraction in aggregate demand. Clearly, the contraction of the Aggregate Demand-side generates an economic slowdown. On the other hand, restriction of workers gathering in the workplace owing to lockdown hurts the production in those sectors which need the presence of physical workers. Note that software or financial service-producing sectors are exempted from such negative supply shock at least in the short-run to medium-run since those works could be easily carried out based on “work from home” anthem. However, reverse migration of workers is an indication of a sharp cut down in production activities. This Aggregate supply shock leads to the situation of stagflation. In recent announcements by the Reserve Bank of India (RBI) to ease liquidity in the system may prove to be counterproductive provided the aggregate supply shock, since the escalation of demand specifically for non-traded products without an adequate increase in production (supply) would only result in inflation. In other words, supply-side management must go hand in hand with demand-side policies. Even if policies are undertaken at the national level by the government, however, conditions may not improve much due to shocks in the external sector through trade in commodities. Most imports are banned which will increase the economic cost of production of import substitutes within the domestic territory based on the theory of comparative advantage in a Ricardian sense. At the same time, exports are hurt due to lower demand from the developed economies. This shows that unemployment and inflation are inevitable in the near future.
Corona Virus has mainly affected raw material export-driven countries. With the Covid-19 virus in the backdrop there will be a decline in new orders, production, and employment with ease in supplier deliveries as demand will be less with a mild decline in the inventories. There has been a huge revenue impact. Long stretches of empty supermarket shelves and shortage of essential commodities are only the visible impacts to consumers of the global chain disruption caused by the Covid-19 pandemic. The uncertainties ahead swing between extremes. As the shortages worsen before they get resolved prices of many commodities could go up for consumers even if laws exist against price gouging. At the same time we should keep in mind that constrained supplies could cause a decline in demand which in turn may end up weakening prices. Risk Management process is not robust enough to cope with the fallout of the coronavirus pandemic. The seemingly relentless forces of globalization and technology with coronavirus in the backdrop will present us with new supply chain challenges and opportunities for further progress in the near future. There is a glimmer of good news as countries globally haven taken preventive measures but the recovery may be fragile.
The crisis due to the global pandemic is likely to be more serious for developing economies compared to the global financial crisis of 2008. This is due to the fact that the global financial crisis leads to lower demand for exports from the developing economies by the developed nations which further lead to contraction of export-based industries and other sectors in the developing nations due to inter-sectoral backward and forward linkage effect, however, the burden of retrenchments in the formal sector was absorbed in the informal sector. In the present situation, the problem of vanishing informal sector is evident since informal sector workers are more vulnerable to the spread of the disease due to the absence of non-regulatory authority and out of direct government control. Moreover, lockdown lead to massive close down of this informal units, thus the informal sector in this present set up is vanishing due to which the shock absorption capacity of developing economy like India is becoming weaker. The global economy is drowning towards deep cycles of economic depression and the aftershocks seem to be long-lasting. The only way of this global shock seems to be international policy coordination in line with the needs of the domestic economies
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

