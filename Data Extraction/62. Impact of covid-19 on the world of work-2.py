
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/estimating-the-impact-of-covid-19-on-the-world-of-work-2/'

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
text = ('''Estimating the impact of COVID-19 on the world of work
COVID-19 an unprecedented pandemic for us but a “Can-be” possibility for great leaders such as bill gates came into action since the rising of the year 2020. The corollary of this pandemic is so prodigious that the analysis by the UN Department of Economic and Social Affairs (DESA) stated that the COVID-19 pandemic is disrupting global supply chains and international trade which in turn could shrink the global economy by up to 1 percent in 2020, a reversal from the previous forecast of 2.5 percent growth. More than 6.6 million Americans filed unemployment claims and the economic downturn is expected to be the worst recession since the Great Depression as stated by the IMF. India is facing its biggest crisis in decades, with a three-week lockdown initially, but extending further in a nation of 1.3 billion people likely to result in an economic recession, millions of job losses and possible starvation among the poor. It can be said that economic contagion is now spreading as fast as the disease itself. On the contrary nature has pressed the reset button, the environment is having a noticeable benefit from this scenario. Water is getting clearer, the air is turning breathable and various such news and pictures can be easily heard and seen on newspapers, telly, and social media platforms on day to day basis. But this is the storyline of the present, what about the future? Are the predictions made by financial institutions and thought leaders going to be true? Or something unexpected is going to happen that will make us think of our human capabilities again?
After this storm of COVID-19, it would not be a surprise to see how digitally every business will grow to provide better digital infrastructure and customer experience along with advancement in technology which we are lacking in this present scenario. The touch screens that we are using already will find its vast applications and would be seen in most of the places including hotels, hostels, and shopping malls. All those processes which are interdependent will take an agile turn resulting in better productivity of goods, having a systematic backup giving a better experience to the consumer even in hard situations. The major focus would be making contactless systems hence Artificial intelligence and machine learning will become the most rapid and in-demand fields. Not only this, but the medical infrastructure and services will also take a boost in the race of developing the systems to handle any situation afterward. The human to machine interaction will increase facilitating fast production, delivery and surveillance inculcating many more. In each and every area the COVID will leave its mark may it be retail sector or may it be the banking sector making them modify or fully change their architectures.
Even the existing systems would be gifted with some modifications, ensuring them to work in a hard environment even with or without human intervention. The Companies will rethink their policies shaping themselves accordingly to manage the resources ensuring safety, agility, and work to be a top priority. The work from home idea will also provide a base for bigger companies ensuring to develop a better infrastructure for their resources and probably giving birth to a new concept of work culture. Various existing industries related to either delivery or travel may adopt the new idea of collaboration with other industries which will result in more options for consumers on their old routine Apps.
New businesses will root out from the existing ventures resulting in head to head competition in the business world but turning into a brownie point for consumer experience and bandwidth of choices. Start-up culture will go hand in hand and expected to take a stupendous growth in the race of providing us a legitimate opportunity to use and feel the real digital world. The various innovations will now take a pace in each field and maybe we will hear some new jargons related to tech in upcoming years. Since the global economy has fumbled, countries will now try to take a different turn to vamoose out of this phase. They will try to make new connections by nurturing a give and take relationship and several business relationships would take an unexpected move making the position of several countries sliding up and down on indexes. Even rival countries may join their hands in this hard time rather than being dismissed totally under predator countries. It would be surprising to see how these scenarios will be handled by the emperor of minds and how their opponents are going to take it as an agenda for the next elections.
Unfortunately, the nature that is at its best will start to face the ill effects after the chart of development and economy will take a pace for exponential growth. It is unfortunate that the inverse relationship between economic growth and environmental destruction exists since carbon emission increases with economic development. As shown by the graph below whenever recession took place the carbon emission decreases along with other harmful gases, turning out as a gift for nature but doom for the financial markets.
But there is a possibility that several start-ups now emerge with the idea of environment conservation collaborating with NGOs innovating and using the best of their capability to reduce the ill effect from the present origin point.
In the culmination we can say this pandemic is like “Blessing in disguise ” producing a massive impact not only on macro but also on micro-economy. The novel virus not only brought some unexpected results but also uncovered some of our loopholes and unlatched new opportunities for growth, forcing us to define ourselves in several new spheres. Quite a while ago we all were chit-chatting about the digital Era, how this is benefiting us, COVID-19 revealed making us realize this was just the beginning of digital Era and a lot is needed to happen in future. It is obvious that not only the Indian economy but the global economy is facing and going to face the aftereffects however similar to a virus, the economy will also rise from deep grounds to a green candlestick with individual collaboration!
The businesses will soon rise again, people will get back to their jobs, vendors will start getting profits and everything will fall into its places but now a megalithic competitive environment will rise which will grow till decades until nature again shows it’s color and we as a human abide with its outcomes, learning some new lessons and keeping this cycle of fall and growth running.
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

