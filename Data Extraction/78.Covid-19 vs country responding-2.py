
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/covid-19-how-have-countries-been-responding-2/'


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
text = ('''COVID-19: How have countries been responding?
“If anything kills over 10 million people over the next few decades, it’s most likely to be a highly infectious virus rather than a war” – Bill Gates
Bill Gates in his TedX address in 2015 pointed out to the predicament of world leaders who failed to invest sufficiently into health infrastructure and he further elaborated on the inability of the prevailing health facilities to accommodate the next pandemic the world would face.
The whole world is grappling with COVID-19- the virus that took birth in the city of Wuhan and spread across the globe, claiming more than half a hundred thousand lives and affecting more than two million people.
With the fear of the virus spreading, the whole world entered into a complete and stringent lockdown mandated by the national governments. This caused economic activities worldwide to come to a standstill except for sectors that provisioned essential goods and services. The international agency-IMF has projected that the global output will contract by 3% in 2020 which is even worse than the 2008-09 financial crises. Moody downgraded the global growth forecast to 2.5% in 2020 from an earlier estimate of 5.3%. The US economy is expected to contract by 2%, Euro-era by 2.2% and China is expected to grow at 3.3% rather than an earlier estimate of 6% in 2021
The national governments have been presented with inevitable challenges of ensuring sustenance and livelihood to masses, improving investor confidence and preventing the economies from falling into a recession in the absence of consumerism amidst the uncertainty about the longevity and future intensity of the pandemic.
The GOI took the decision to enter into lockdown in the preliminary phase of the spread of the virus. The Indian economy in the fiscal year 2019 was already hit by severe NPA crises. It witnessed a downward spiral and grew at a mere rate of 5% as against 6.8 % in 2018 as per the statistics released by the NSO-National Statistical Organization. The manufacturing and construction sectors were negative outliers. The government in order to recuperate announced a plethora of decisions. The government cut the corporate tax from 30% to 22% and for manufacturing enterprises from 25% to 15%. In order to strengthen the position of public sector banks, the government announced the merger of 10 public sector banks into 4
The government also announced a fiscal stimulus of 1.7 lakh crores INR stepping over its fiscal deficit target in order to drive the economy out of the downward spiral and get it on the path of growth, giving lubricant to an economy facing liquidity crunch with heightening NPA.
But the Coronavirus outbreak halted the economic activity and posed government the issue of fulfilling the basic survival needs of workers in the informal sector in the wake of massive urbanization. The daily wage earners tried to walk down to their native places in huge numbers posing immense difficulty for the government. Apart from that the domestic equity market indices- the BSE Sensex and NSE Nifty crashed which reflected the erosion of investor confidence and reflected stealth of bearish sentiment around the world.
All this amalgamated together posed a serious question on the survival of a huge chunk of the Indian population and invited a lot of criticism for the government.
This is when the Central NDA government announced a fiscal relief package: The Central government announced a fiscal stimulus of additional INR 1.7 Lakh Crores that constitutes approximately 0.8% of India’s GDP under Pradhan Mantri Gareeb Kalyan Yojna. This fiscal stimulus is used to target the underprivileged section of the society instead of conventional indirectly lubricating the economy through deficit financing.
The government under the NFSA-National Food Security Act has decided to provide an additional 5 kg of Indian staple wheat and rice and 1 kg of pulses every month to people with ration cards. The government is expanding and accelerating the dissemination of unilateral transfers under various government schemes.  Under PM Kisan Scheme an Indian farmer receives payment of INR 6000 in three installments every year. Out of 6000 the GOI had disbursed INR 2000 immediately. The government has made a provision of providing 2 installments INR 1000 for the next three months those already availing old ages, women, or disability pension. The women who are Jan Dhan account holders are to receive INR 500 for the next three months. The government assured the availability of cooking gas by providing 1 LPG cylinder for the next three months to 8.3 crores people covered under Ujjwal Yojna.  The government in order to prevent the deterioration of and encourage entrepreneurial spirit in already discriminated rural women the government made a provision for Women SHG’s where they can receive collateral-free loans up to INR 20 Lakhs under NRLM (National Rural Livelihood Mission). The government also enhanced payment under MGNREGA from INR 180/day to INR 202/day.
The government has also taken upon itself to provide for the EPF contributions of 12% each of the basic salaries of both employees and employers for the next three months. In order to ensure household liquidity the employees can withdraw from their EPF accounts balance up to 75%. The GOI is also providing insurance coverage if INR 50 lakh for a tenure of 3 months for about 22 lakh health workers in government hospitals inclusive of ASHA workers, medical, sanitary workers, and paramedics.
The finance minister Nirmala Sitaraman also pointed out that increased money supply combined with the dollar swap with the USA will help curb the inflammatory effect also.
Shaktikanta Das, the RBI chief announced a monetary relief package for the economy that seemed to regain investor confidence as the national stock indices rallied. The following are the elements of the package:
Despite tourism, hospitality, domestic trade, airline industry taking a hit, India was presented with quite a few opportunities as well.
India came as a messiah to export hydroxychloroquine to the USA, followed by Indian pharmacy companies- Lupin, Dr. Reddy’s Laboratories, Strides Pharma’s plants in India, and Biocon’s plant in Malaysia being given clearance by US FDA-US Food and Drug administration. This gives massive export opportunities to these entities, improving their growth prospectus. The crash of crude oil prices caused by falling out of negotiations between OPEC and Russia creating supply pressure along with lowered demand enabled India to lower its import bill. According to a report by ICRA every $10 decline in crude oil price saved India $15 billion.
Not only India but countries like the USA, Japan, UK, and Germany have announced massive fiscal stimulus packages to equip their economies to come out of this phase of decelerating economic activity.
US announced a fiscal package of $2 trillion with is approximately 9.3% of its GDP. This relief package is the largest relative to the GDP is the largest of the kind in the Modern American History. It is way larger than $800 billion assistance provided in 2008. Germany introduced largest fiscal stimulus in relation to GDP.
The following graph indicates the fiscal stimulus as a proportion of GDP of respective countries
Workers in the US whose annual incomes are up to $75000 will receive $1200 in direct payments as well as $500 for every offspring. America being America came to the rescue of businesses by allocating $500 billion to businesses and local government bodies.
It announced various sectorial packages-$50 billion for passenger airlines, $8 billion for cargo carriers, $17 billion for businesses involved in maintaining national security. Taking care of the small industry sector, the US government committed itself to devote $350 billion to the working capital requirements of these entities. The government further allocated $150 billion for the health care sector, $45 billion for the disaster relief fund, $31 billion for the education sector, $27 billion for research and development, and $15.5 billion towards the food stamp program.
. Republican President Donald Trump has been under a lot of criticism lately. The tackling of the COVID-19 crises by America could either relinquish Trump administration or them another tenure. But so far the situation seems critical with the USA registered with the maximum number of cases
China- the epicenter of the pandemic has been facing a lot of backlash from world powers like the USA, UK, France, and Germany; each country asking for monetary compensation. The US even alleged that China has conducted low-frequency nuclear tests while the whole world is anguished. China has also been frowned upon for acquiring a stake in various companies like-The PBOC enhancing its claim in HDFC to 1% and many other entities amidst panic selling. China also capitalized on low crude oil prices by creating massive buffers. It further exported PPE to various countries, capitalizing on the minutest of opportunities available.
This is the time to be prudent and try to remain optimistic as is illuminated by Charlie Munger,-“I would ……when the worst typhoon that’s ever happened comes. We just want to get through typhoon and we’d rather come out it with a whole lot of liquidity. …”
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

