
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/how-will-covid-19-affect-the-world-of-work/'

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
text = ('''How will COVID-19 affect the world of work?
Every prophet of doom, unless he also happens to be a psychopath, hopes that his predictions will not be borne out. This is also true for the epidemiologists and virologists who have been warning the world since January that the novel SARS-CoV-2 virus poses a severe threat to public health around the world. The name Coronavirus itself, is vicious enough to terrify those who otherwise are fearless in the face of any ball game. Tracing back the origins of the now pandemic, we find that the coronavirus emerged in Wuhan, a city of 11 million people in China’s Hubei province, in late 2019. Cases of the disease grew by several thousand per day in China in late January and early February, the peak of the epidemic there. The number of infections appearing each day has since plummeted in China, making the outbreak a global pandemic that has engulfed over 214 countries around the globe.
Besides being a health crisis, the virus has already brought a major downfall in the world economy with the IMF estimating the coronavirus-induced global downturn to be ‘way worse’ than financial crisis. As the coronavirus spreads across the globe, it appears to be setting off a devastating feedback loop with another of the gravest forces of our time: economic inequality. Research suggests that those in lower economic strata are likelier to catch the disease owing to their inability to meet the rising drug prices. According to the International Labour Organization(ILO), COVID-19 will have far-reaching impacts on labor market outcomes. Beyond the urgent concerns about the health of workers and their families, the virus and the subsequent economic shocks will impact the world of work across three key dimensions.
Initial ILO estimates point to a significant rise in unemployment and underemployment in the wake of the virus. Based on different scenarios for the impact of COVID-19 on global GDP growth, preliminary ILO estimates indicate a rise in global unemployment of between 5.3 million (“low” scenario) and 24.7 million (“high” scenario) from a base level of 188 million in 2019. The “mid” scenario suggests an increase of 13 million (7.4 million in high-income countries). Though these estimates remain highly uncertain, all figures indicate a substantial rise in global unemployment. For comparison, the global financial crisis of 2008-9 increased unemployment by 22 million. While self-employment does not typically react to economic downturns, it acts as a “default” option for survival or maintaining income – often in the informal economy. For this reason, informal employment tends to increase during crises. However, the current limitations on the movement of people and goods may restrict this type of coping mechanism.
Labour supply is declining because of quarantine measures and a fall in economic activity. At this point, a preliminary estimate suggests that infected workers have already lost nearly 30,000 work months, with the consequent loss of income (for unprotected workers). Employment impacts imply large income losses for workers. Overall losses in labour income are expected in the range of between 860 and 3,440 billion USD. The loss of labour income will translate into lower consumption of goods and services, which is detrimental to the continuity of businesses and ensuring that economies are resilient. Working poverty is also likely to increase significantly. The strain on incomes resulting from the decline in economic activity will devastate workers close to or below the poverty line. The growth impacts of the virus used for the unemployment estimates above suggest an additional 8.8 million people in working poverty around the world than originally estimated (i.e. an overall decline of 5.2 million working poor in 2020 compared to a decline of 14 million estimated pre-COVID-19). Under the mid and high scenarios, there will be between 20.1 million and 35.0 million more people in working poverty than before the pre-COVID-19 estimate for 2020.
Who are particularly vulnerable?Epidemics and economic crises can have a disproportionate impact on certain segments of the population, which can trigger worsening inequality. Based on past experience and current information on the COVID-19 pandemic and insights from previous crises, a number of groups can be identified:
The extent and severity to which the coronavirus pandemic will impact the fight to end extreme poverty is still unknown, but it is expected that the crisis will devastate the world’s most vulnerable people including the world of work. The virus is already disproportionately impacting the poor in wealthy countries, where the most known cases are concentrated. Experts are urging the world to prepare to lend extra support to low-income countries to address the pandemic. COVID-19 cases are more likely to go undetected or to be under-detected in developing countries that have fewer resources available to tackle a pandemic. Countries with large poor populations including Brazil, India, Indonesia, Nigeria, and Pakistan have confirmed few cases, but have been slow to respond to the threat, according to NPR. Preventative care and health education are less accessible to low-income people who are more likely to have pre-existing conditions, catch COVID-19, and die from it. People living in poverty are also more likely to hold insecure jobs and cannot afford to stay home sick from work.
In times of crisis, International Labour Standards provide a strong foundation for key policy responses that focus on the crucial role of decent work in achieving a sustained and equitable recovery to mitigate the impacts of COVID-19 on the world of work. These standards, adopted by representatives of governments, workers’ and employers’ organizations, provide a human-centered approach to growth and development, including by triggering policy levers that both stimulate demand and protect workers and enterprises.
Policy responses should focus on two immediate goals: Health protection measures and economic support on both the demand- and supply-side of the associated cause.
In such challenging times, many countries have come forward to protect their workers in the workplace with decisive measures to combat the spread of the disease, while ameliorating its pernicious effect on the economy and labor market across the three policy pillars: protecting workers in the workplace, stimulating the economy and labor demand, and supporting employment and income. Some of the policies include Working arrangements, including telework; Occupational Safety and Health (OSH) advice; Expanded access to paid sick leave; Prevention of discrimination and exclusion. While these measures will no doubt help to contain the pandemic, to respond to the emergency needs it has generated and to pave the way to a gradual recovery, it is clear that more needs to be done. Past crises and the experiences of countries, which have reacted too late in the context of the current COVID-19 crisis, show that preparedness and early action is critical.
The coronavirus pandemic is a test. It’s a test of medical capacity and political will. It’s a test of endurance and forbearance, for believers a test of religious faith. It’s a test, too, of a different kind of faith, in the strength of the ideas humans choose to help them form moral judgments and guide personal and social behavior. The epidemic forces everyone to confront deep questions of human existence, questions so profound that they have previously been answered, in many different ways, by the greatest philosophers. It’s a test of where all humans stand. What is right and what is wrong? What can individuals expect from society, and what can society expect of them? Should others make sacrifices for me, and vice versa? Is it just to set economic limits to fighting a deadly disease? How long will we stay that way? And as the epidemic grows worse and brings the disease within fewer degrees of separation for everyone, we may well find that the notion of loving thy neighbor as thyself becomes far more potent.At last I would like to conclude by saying that COVID-19 is something we all face as a community, and thus something that we have to solve as a community, not with weapons, but with goodwill and common efforts.
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

