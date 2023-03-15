
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/environmental-impact-of-the-covid-19-pandemic-lesson-for-the-future/'
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
text = ('''Environmental impact of the COVID-19 pandemic – Lesson for the Future
The Covid- 19 pandemics forced factories to shut down, flights getting canceled and a massive decrease in the global economy, with a significant decrease in Green House Gases (GHG) in many developed and developing countries.
The SARS- CoV2 came into the spotlight in December 2019 and has impacted most of the countries till then. Nearly 131 million peoples were infected worldwide and resulting in deaths of around 2.9 Million according to World Health Organisation (WHO). Most of the countries dealt with the new virus by imposing strict lockdowns and social distancing to control the spread of the virus. These policies caused adverse effects worldwide. One of the most important impacts of the Covid-19 Pandemic is on the environment.
There have been few positive impacts on the environment due to lockdown like, air pollution has decreased dramatically, as people were asked to stay in their houses due to the lockdowns. There has also been a sharp decline in environmental noise. Environmental noise can be well defined as the unwanted or harmful outdoor sounds caused by human activities like noise emitted by road traffics, air traffics, rail traffics, and industrial activities. It is one of the most important challenges in the modern era as noise pollution can cause adverse effects on humans as well as harm many animals too. The imposition of lockdown and quarantine by various nation’s governments has caused people to remain back at their homes. Because of this, the movement of people from one place to a different place has reduced significantly and the use of personal and public transport has also decreased. Due to all these changes, the environmental noise generated in most of the cities has dropped substantially.

Water Pollution at beaches have reduced significantly and many animals were spotted back in the cities, but the covid-19 virus has also generated many negative and indirect effect on the environment.
To begin with, some of the developed countries have halted their sustainability program during the pandemic. In the United States and in many other European nation-States, waste recycling plants have been suspended in many municipalities and cities due to the concern of the virus getting spread at the recycling centers. This has resulted in an increase in the use of single-use plastic bags instead of the re-usable bags by many leading restaurants, firms, and corporations. For instance, Starbucks, a leading coffee company during the month of March 2020, has announced a short-lived ban on the utilization of recyclable and reusable cups.
Furthermore, with most of the people staying indoors because of the lockdown majority of the department stores, shops, restaurants, and food outlets are closed, making an online purchase and food delivery are quite high. This has created more consumption and demand for fossil fuels due to the transportation and mobility of these goods to each individual. There has been an enormous upsurge of medical waste- because most of the products employed by healthcare professionals are usually single-use items that can be used only once before they are disposed of. Some of these wastes include used masks, personal protection kits, and gloves. For instance, nearly 200 tons of medical waste were generated during the peak of the pandemic breakout in Wuhan, China. This is 50 tones more than the average waste generated before the outbreak in Wuhan.

These organic and inorganic wastes generated due to the policies crafted by the government takes a heavy toll on the environment and can cause environmental issues like water pollution, air pollution, soil erosion, and can harm the local flora and fauna.
The demand for masks during the pandemic has become skyrocketing but the materials required for the production of these masks are highly dangerous for the environment as they are generally composed of non-woven fabrics. Polyester, Polystyrene, polyethylene, and polycarbonate, are some common materials used for the surgical mask with density lying between 20grams to 25 grams/sq. meter. These materials are mostly resistant to liquids and are plastic-based products with a really high afterlife after being discarded. If not treated properly without discarding, they end up filling the landfills and the oceans making it dangerous to aquatic lifeforms. For example, recently the environment in Hong Kong has started degrading drastically during the pandemic with the accumulation of these clinical wastes.

With recycling plants on hold, piles of small mountains of wastes and their depositions at open areas are formed due to the increasing number of unrecycled wastes generated every day. This makes the surrounding more vulnerable and also creating a high risk of air pollutions as the dumped open wastes decays into Methane (CH4), a greenhouse gas, hence increasing the risk of global warming too.  Not only the surroundings are getting affected by these careless methods, but the local people are also getting affected. If the excess methane gets accumulated in the Earth’s atmosphere due to piles of unrecycled wastes, this could result in increases in Earth’s average temperature and can be harmful to future generations.
Many protected and endangered flora and fauna are facing much greater risks due to the imposition of the lockdown. Many countries under lockdown have made people stay inside their homes. The employees, NGOs, and volunteers working in these protected areas like National Parks, Marine Conservation Zones, and wildlife sanctuaries are made to stay at home making these protected and endangered flora and fauna unattended. This has also led to an increase in many illegal activities like wildlife hunting, illegal deforestation, and fishing activities due to the absence of these people. The prohibition of eco-tourism has also led to a significant financial drop in the economy of these protected areas.
Some nations like China have asked the local authorities and native government bodies to increase the amount of disinfection routine, mainly to increase the dosage of chlorine in the wastewater treatment plant to prevent the spread of the COVID-19 virus. However, according to WHO (2020), no solid evidence has been found on the survival of the virus on the lifespan of the virus in wastewater as well as in drinking water. Despite the fact that excessive amounts of chlorine in water can cause harmful problems and issues associated with people’s health like bladder cancer and it also damages its cells.
The COVID-19 is a reminder that the health of the planet is also linked to the health of humans. Evidence proves that the virus is zoonotic, meaning it can be transmitted between people and animals. They are accountable for seventy-five percent of all emerging infectious diseases in the World. With the Virus infecting millions of people every day across the globe, Various governments and agencies’ top priority is to regulate the spread of the virus by shifting the spotlight on the management and treatment of wastes (especially the clinical and medical waste).  Likewise, at the same time, we as responsible individuals need to step up and follow the necessary guideline and precautions for the disposal of the waste and medical gears.

In Spite of various data showing that the pollutions have reduced significantly during the pandemic, History has witnessed a rise in pollution during any “post-financial crisis”.  A similar case was observed during the 2008 financial crash – although there was a temporary decrease in emissions of 1.3% was observed, but as the economy recovered in 2010, emissions were at an all-time high. After all, only through sheer mutual empathy and goodwill that the world will emerge stronger after this global pandemic. To prevent future outbreaks, we must address regularly the threats to the ecosystems and wildlife, including habitat loss, illegal trade, pollution, and climate change as human life depends on Earth’s life.
If you live near a spacious outdoor area, like the desert or an empty road lined with trees and you realize it’s the only safe, surface-less space to take a walk in, then you begin to realize the beauty of nature. The point is not to remain indoors, but to avoid being in close contact with others. When you do leave your home, whether it is for a walk in the desert or a run on your street, make sure to wipe down any surfaces you come into contact with, avoid touching your face, and frequently wash your hands.
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

