
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/covid-19-how-have-countries-been-responding/'

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
The COVID-19 pandemic is impacting communities all over the world.
With 149 offices in countries worldwide, WHO is leading the global effort to support countries in preventing, detecting, and responding to the pandemic. WHO is also monitoring the response: 143 countries have now their own COVID-19 response plans, and almost all (194) countries have adopted public health and other response measures based on WHO’s guidance.
The response also covers the procurement of supplies. As of 2 March, WHO has bought and shipped 1.5 million coronavirus test kits and close to 800 000 face masks across the planet, delivering personal protective gear to more than 70 countries. At the same time, more than half of WHO’s personnel are responding on the ground by providing real-time updates, expertise, and coordination as governments, humanitarian agencies, and the public race to respond.
Countries are adopting different ways to contain the spread of coronavirus but there is no one-size-fits-all approach.
Towns and cities have been locked down and large gatherings banned.
Restrictions have been imposed on travellers from hard-hit areas, such as China, Italy and Iran.
Major sporting events, carnivals and events have been postponed or cancelled.
The COVID-19 coronavirus has now spread to every continent except Antarctica, challenging health systems and governments everywhere. Although the vast majority of the almost 90,000 cases around the world are in China where the virus originated, 64 different countries* are now affected.
For most, the virus represents a mild health issue, but for vulnerable members of society the consequences can be more serious. Containment remains a priority for all countries but there is no one-size-fits-all approach to tackling the spread of the disease.
France
The French government has advised its citizens to abandon the customary “bise” greeting – involving kissing each other on the cheek – in a bid to slow the spread of COVID-19.
Public gatherings of more than 5,000 people are also off limits, resulting in the cancellation of events like the Paris Half Marathon. Following the decision, the Louvre museum in Paris closed its doors to the public to mitigate the threat of infection posed by visitors arriving from different parts of the world.
Iran
As the Middle East’s worst hit country, nearly 3,000 cases of COVID-19 have been reported in Iran, including more than 20 lawmakers. The country’s parliament has been suspended indefinitely and MPs have been asked to cancel all public meetings.
Iran’s death toll is the third highest, after China and Italy, and medical supplies are running short. Exports of face masks are banned for three months, while Iran’s factories produce new supplies for local people.
Germany
German Health Minister Jens Spahn has declared coronavirus a ‘worldwide pandemic’, something the World Health Organization has not concluded at this point. The government has banned the export of medical equipment, as Spahn said the virus there had not yet reached its peak.
United States
California has declared a state of emergency after the first death in the state, which brought the U.S. death toll to 11. The move follows Washington and Florida both declaring a state of emergency, with 10 of the deaths in Washington state. The government is preventing entry to anyone who has visited China in the last 14 days and has expanded testing nationwide.
Switzerland
Precautionary measures are in place in Switzerland, where gatherings of more than 1,000 people have been banned, forcing the cancellation of annual events like the Basel Carnival and the Geneva International Motor Show. Interior Minister Alain Berset has also advised against using the country’s customary three-kiss greeting
Austria
Authorities in Austria imposed a ban on trains travelling on key international routes to and from Italy, such as the Brenner Pass. The move followed two suspected cases of coronavirus discovered on a train heading from Italy to southern Germany, which later tested negative. The temporary ban has now been lifted, allowing scheduled rail services between Austria and Italy to resume.
Italy
Italy has shut all its schools and universities for 10 days, as the government also banned public conferences and cultural events to curb the spread of the virus, which has already killed more than 100 people.
Some towns in northern Italy’s Lombardy region are in lockdown. Restaurants and businesses are closed, threatening to plunge the country into recession.
Curve of confirmed Covid-19 cases outside of CHINA
China
At the epicenter of the outbreak, China has adopted aggressive measures to contain the virus, including city lockdowns, travel restrictions, extending school breaks and closing down theatres, sporting events, and other public venues. Infection rates continue to increase, but the rate of increase has slowed.
Hong Kong
Hong Kong’s border with mainland China has been closed, preventing visitors from entering the territory. Without the throng of global tourists that usually flock to Hong Kong, the economy has been hit hard. Schools are closed until April, and many flights in and out have been restricted or canceled. Hong Kong’s recently unveiled budget included a government payment of more than $1,200 for each resident to help ease the economic pain
Japan
Japan’s Prime Minister Shinzo Abe has called for all elementary, middle, and high schools to close until late March, impacting millions of students. The threat posed by the virus could jeopardize the Tokyo 2020 Olympic Games, due to be held in the summer, although no decision to cancel the event has been announced.
Saudi Arabia
No coronavirus cases have been detected in Saudi Arabia, but there have been some in regional neighbours like Kuwait and Bahrain. Authorities have barred entry to the kingdom for foreign pilgrims from 25 countries, preventing visits to Islam’s two holiest sites – Mecca and Medina.
United Arab Emirates
Ferry services between the UAE and Iran have been suspended and all commercial ships must provide health statements for crew members 72 hours before arriving in the country’s busy ports.
South Korea
outh Korea has the most cases of any nation outside of China. Strict self-isolation requirements are in force throughout the country, with fines or a potential prison sentence awaiting anyone found violating the rules.
After military personnel tested positive for the disease, planned annual joint military exercises with US forces have been put on hold.
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

