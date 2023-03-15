
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/impacts-of-covid-19-on-streets-sides-food-stalls/'


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
text = ('''Impacts of COVID 19 on Streets Sides Food Stalls
Petty Entrepreneurs like Streetside Food Stalls form an essential part of an economy that plays a significant role in balancing the development of a nation’s economy. The importance of street food vendors not only helps in increasing the country’s per capita income but also helps to make a living for the unemployed in society with a limited investment. Most of them are the sole breadwinners for their families. So, with the Covid-19 wreaking havoc, the uncertainty of the lockdown policies increased the concern of people over hygiene and the lack of investment to upgrade their businesses to provide better hygiene facilities that can take away food from their mouths.
 Various states have provided relief packages and support but, in a country, as populous as ours, it has hardly been appropriately implemented. Delhi, for instance, recently announced a rupee 500 million stimulus package to nearly 500,000 rupees, recognizing the severe consequences of the loss of livelihoods. The seller’s intended remedy is a credit loan that provides all sellers with an initial working capital of Rs 10,000, but this is not enough. Instead of loans, the government should consider converting them directly into income benefits, cash subsidies to secure a livelihood in order to initiate income-generating activities on a regular basis. Sellers need income support to get back to work, and if they can’t, they will never be able to repay the loan amount. In an ever-changing crisis, organizations need to take a step forward and ensure that providers are provided with the resources they need to make a living. Pandemics have had a vast economic impact on every sector, but the most vulnerable sectors, such as street vendors, suffer the most. Decreased income can affect most necessities, such as payments to family and yourself. Due to a lack of fixed salaries, stalls cannot quit their jobs. If possible, it might be only a few days, unlike regular employees. For them, the risk of infection is far less frightening than hunger.
Unlike many of the businesses, most of these food vendors lack the infrastructure to go online and sell products through platforms like Swiggy or Zomato.
“Regret not going digital before,” says Dilip Das, a Street side food stall owner at Sector V, Kolkata. Dilip Das used to run a small streetside food stall that provided meals to all the employees around the area which is filled with multinational companies. It was a booming business till the arrival of Covid. Then with the lockdown arriving, almost all the companies opted for Work from Home policies. Dilip tried to start deliveries, but with a lack of support and fear of infection, that business ended before it even started.
Citing similar reasons, Swapan Mondal, who runs a south Indian food stall at Park Street said that he did know how to operate a smartphone. And since sales were good, he did not care about learning either. But with the implementation of lockdown which took place within 4 hours, people like Swapan barely had time to cope with the changes.
Furthermore, the Covid 19 pandemic made people realize how important trains are in our life. With most of the train network suspended, the entire supply chain was disrupted. Hundreds of quintals of food products rotted away, awaiting the logistical issues to fix. Many businesses could not afford to run without proper supplies as well.
To handle such issues, vendors organizations may consider the following solutions:
Promotion of livelihood for all vendors, including sellers of non-essential goods
The impact of COVID19 was extremely severe for informal workers who ran out of capital and income and to try to feed themselves during this long embargo. Sellers need to be able to resume sales in order to survive, and governments need to take steps to reopen the market and bring sellers back to the business.
Reopening of Markets keeping in mind social distancing and hygiene
India has different types of traditionally crowded markets, even weekly markets (fresh food, cooked food, and essentials) and daily roadside markets. These markets have to reopen with the need for social distance in mind, and the government needs to publish guidelines for that. In the future, sales areas will also need to be designed with social distance and the need for adequate sanitation (running water, wash areas, toilets) in mind. Authorities need to work with the Town Bending Commission (TVC).
Provision of direct support that is devoid of existing registration requirements
Once the restrictions are lifted and the impact of Covid 19 is somewhat manageable, sellers who have been at home for months will need direct income benefits to help them to return to work. The government initiative is a welcome move, but not in terms of the type of bailout and eligibility is enough. In addition, there are very few registered providers in India, so government relief and support must be separated from the very strict registration requirements. In Delhi alone,  of the approximately 300,000 street vendors, only approximately 13100 have some sort of professional ID. If all types of cash grants or livelihood support standards relate to job identification by the state,  the government must also accept registration with workers’ organizations/unions on behalf of government-issued sales badges.
Ensuring social distancing and proper sanitation and hygiene at sites of businesses
The government and municipal corporations need to take the initiative and running water, and soap/sanitizers need to be provided for street vendors at their place of work. Moreover, the streetside food vendors should work with food safety authorities in the country to train themselves in different ways to maintain hygiene whilst working.
Taking steps to spread awareness of different government initiatives and reliefs
Many small businessmen and food vendors do not even know their rights and the benefits they can claim. If they did, many still have no idea about how to claim them. Proper awareness camps need to be set up to help these small-time food vendors to know how they can get assistance and help from the government and return back to the business.
When Covid was an unknown word, these shopkeepers and vendors brought the taste to our mouth, gave us memories to live for, filled our bellies, and quenched our thirst when we needed them. It’s time to provide them with a hand now when they need it.
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

