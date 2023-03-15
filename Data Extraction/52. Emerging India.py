
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/changing-landscape-and-emerging-trends-in-the-indian-it-ites-industry/'



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
text = ('''Changing landscape and emerging trends in the Indian IT/ITeS Industry.
The COVID 19 pandemic has reshaped the fundamental fabric of our world. Initiating in China in late December 2019, the virus has since then spread to every corner of the world, infecting millions in its wake. Countries around the world have responded by shutting down their economies and advocating lockdowns of varying degrees. India has arguably had one of the most stringent lockdowns of all countries, restricting the movement of people, goods and effectively transforming every city into a ghost town.
Due to such stringent restrictions on movement, the economy has received a fatal blow. The virus arrived in India at a time when the country was already in strife, dealing with a slowing GDP growth rate and an ever-increasing fiscal deficit. The lockdown has simply exacerbated the economic predicament; pushing multiple businesses and industries to a moribund state. Other countries have also faced a similar predicament, particularly in the US and Europe, both being hotspots for virus transmission.
However, upon taking a closer look at the global economy, there seems to have been an upward mobilization in some of the sectors in the IT domain. These sectors seem to have exorbitantly increased their scale of operations and have overall reaped hefty profits. So what is it that differentiates these areas from the rest and what can we learn from them?
If we look at the evolving IT/ITeS industry around the world, the domains that have flourished in this strenuous time all have had one key characteristic in common; flexibility. These sectors have adopted a more flexible approach to dealing with the lockdown. Be it flexible working procedures or adopting newer, more advanced ways of working in line with the requirements of our time, they haven’t shied away from changing the rules of the game. Let’s look at some of these areas within the tech domain that have been successful in conquering the lockdown.
Lockdowns around the globe have confined people inside their homes, with nothing much to do of significance. As a result, social networking sites have reported a gigantic rise in user engagement over the lockdown period. Reportedly, there has been close to a 50% rise in daily text transmissions across messaging platforms such as Facebook, Instagram, and WhatsApp in the hardest-hit economies. Twitter has reported a 23% increase in its user base as compared to last year. The data makes it evident that social media platforms are here to reap the benefits of a lockdown situation.
Video consumption sites such as YouTube, Instagram, and TikTok are in line to reap massive profits. It’s no wonder that viewership on these platforms has increased manifolds due to lockdowns. These platforms serve to be a distraction from the tragedies of the real world in such stressing times and offer an easy escape. Be it rumors regarding the cure for COVID 19, gossips, or rap battles; these platforms provide viewers with their much-needed dose of distractions. As they say, “ignorance is bliss”.
The lockdown has proven to be a miracle for data-mining companies. With such an exorbitant increase in the number of hours spent online globally, these firms now have access to significantly larger sets of data than they otherwise could ever have. Widespread implementation of data-mining has been seen recently with most countries opting for Government-regulated surveillance applications that monitor your movement at all times. Moreover, these applications have access to your personal information as well as medical records. Countries such as South Korea and Singapore are at the helm of relying on such means to control the outbreak.
Although noble, such initiatives do raise a red flag with regards to user privacy. The public has been provided with very little information about how these applications operate and what goes on behind the scenes. This creates an atmosphere of obscurity that is frankly harmful to the premise of a democracy.
Cloud Computing
Cloud computing has been on the rise for the last couple of years. However, the pandemic has made it the de facto king in terms of computational services. With an increasing reliance on remote working, the ability to store data in one secure location has turned out to be critical.
The cloud computing services such as Amazon Web Services, Microsoft Azure, and Google Cloud have reaped hefty profits in the process. The emerging landscape in the tech world seems to suggest an upward trend in reliability on these services as far as large scale data storage and computing is concerned.
Social distancing measures have made it practically impossible for people to physically go shopping for groceries and other necessities. Moreover, a majority of retail outlets in cities have reported negligible engagement, for the fear of transmission of the virus. E-commerce sites have been the real winners in this predicament of global consumerism.
With high standards of hygiene and at-home delivery, these sites have provided an easy alternative to retail stores. Other similar businesses include food delivery apps and grocery sites that are also flourishing today amidst the pandemic. Recent weeks have made it evident that E-commerce websites are well on their way to abolish traditional shopping methods.
Online streaming platforms consist of a relatively newer approach to entertainment; having only been in existence for the past couple of years. They had already, however, established a name for themselves before the lockdowns. Sites such as Netflix and Amazon Prime had been dominating a chunk of the market share and competing heavily with movie theatres for new releases, for a while now. Now with the lockdown practically deleting theatres and cable TV from the collective psyche of the entertainment industry, they are well on their way to becoming the platform of choice as far as entertainment for the masses is concerned.
This has already become evident, with Zee5 India reporting a whopping 80% increase in subscriptions and a 50% increase in the time spent on their site, within the first couple of weeks of the lockdown itself. Another such platform MUBI reported a 28% rise in its viewership in India only a week after lockdown commenced. These facts are a clear indication of the exponential growth that is in store for these platforms, which is unlikely to change anytime soon.
The idea of working from home is all set to be the new norm in our professional culture, which requires an easy and effective way to communicate over the internet. Not just audio or text, but video conferencing is the need today. Hence it’s understandable why multiple video conferencing services are set to reap massive profits off of the new norm.
Arguably the most popular of these, Zoom has reported a 130% increase in its price share since the beginning of 2020. Another such platform known as the Microsoft Teams collaboration suite, operated by Microsoft, has reported a whopping 12 million increase in its user base in the very first week of the lockdown in the US. This data is only likely to have an upwards trajectory considering the increasing pace at which industries are going online.
The Indian Picture
The Indian IT landscape has also undergone massive cultural changes. According to Rajesh Gopinathan, chief executive officer and managing director of Tata Consultancy Services Ltd., the “work from home” model of operations is here to stay. He also claimed that the firms only requires a 25% workforce to be physically present for any project at any of the locations.
Tata Consultancy Services which is one of the major players in the Indian IT landscape has shifted almost 90% of its workforce into a remote borderless workspace model, and other smaller firms have followed suit. What it tells us is that we’re headed in a direction we’re unlikely to return from. Remote working appears to be the modern protocol for operations and is unlikely to change anytime soon.
However, a major challenge often overlooked in such circumstances is that of an adaptable culture. The culture of workspaces has to be drastically altered and members of the workforce have to dive headfirst into the new way of doing business. This is critical to ensure that the industry can combat the pandemic and recover gloriously.
The lockdown has presented us with a stark contrast unlike any other we’ve witnessed. Where on the one hand, the industries mentioned above seem to be exhibiting exponential growth with their people thriving and having every possible resource at their disposal, the unorganized sector and the MSMEs (Micro, Small and Medium Enterprises) seem to be at the crux of death.
The sector that consists of people for whom having guidelines to work from home is irrelevant as that was never really an option; people with no fixed wages or social security to fall back on during this time of crisis. As a result of the lockdown, several of these MSMEs have effectively become inoperable and have resorted to massive layoffs just to survive. This has resulted in thousands of workers losing their jobs with nowhere to go to for refuge. A chunk of these people includes migrant workers who have lost their jobs and are left with little to no avenue to feed themselves. Not receiving significant support from the government, these people have been initiating treks back to their villages. What is witnessed is a massive influx of migrant workers flooding the highways, walking weeks to find shelter and food. Truck drivers responsible for delivering goods have also resigned stating that they have no means to eat because of the closure of every food joint on the highways. Both these realities present a stark contrast and expose a deep crack in our societal framework. One I reckon won’t heal anytime soon.
The COVID 19 pandemic has restructured our social fabric in a way that is unlikely to change anytime soon. The areas that we’ve discussed are likely to become the flagbearers of the digital revolution of the decade. An upward mobilization with regards to massive scale movement of industries online seems pertinent today being led by these domains in the IT industry. An unfortunate consequence of it is that a majority of the MSMEs may soon cease to exist, having been replaced by some of the other alternatives that we’ve discussed. It seems to be a never-ending circle where the ones who are making a profit out of this situation will continue to do so, while the others will be washed out from our collective memory.
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

