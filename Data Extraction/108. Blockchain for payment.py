import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/blockchain-for-payments/'


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
text = ('''Blockchain for Payments
Reconciling with the financial realities of an MBA education may come too late for many. There’s a paper-thin or let’s say “paper-money thin” difference between having a tight budget and being seriously stretched thin for money. Dangerously enough during the second month of my MBA, I was treading on the prospect of the latter. It struck me that I could find my savior in an uncle who lived in San Jose and was the type who always saved for a rainy day. With online payment avowedly ubiquitous and simple, it seemed like a no-brainer to me. One tap on an app and I would be rid of this predicament I had found myself in. Quickly enough, he initiated a fund transfer via Xoom, a PayPal service. I was dismayed to learn it would take all of three days for the money to finally reach me!
One would believe that in the era of globalization and high-speed internet payments would be fulfilled in a trice. But contrary to expectations, I learned that most cross-border payments are subject to numerous go-betweens and channels rendering them behindhand especially in some parts of the word. A complex series of interactions involving banks, merchants and other parties underlie the seemingly non-complex act of initiating payment on the user-friendly one quick Interfaces. Anyone who has transferred money between countries recently realizes the process has evolved little in decades. Too slow, untrustworthy and too expensive are the gripes. And as more of us become plugged into the globalized economy, the scale of the problem is surely going to skyrocket. This begs the need for if a 21st-century digital solution to cross-border payments which could be faster, less expensive, more reliable and transparent.
Consider the case of Ashok Chandra, a mid-tier textile exporter. His business entails frequent wire-transfers to Dubai and Indonesia. These wire-transfers can take up to 3 days causing Mr. Chandra to worry about delays in addition to working capital constraints. He feels left in the dark regarding the standing of his dealing and out of pocket high fees
Such wire-transfers are often a major deterrent to the volume of trade. Small to medium-sized enterprises, above all, suffer from payments difficulties. It may take weeks for a cross-border payment to settle, which may place the brakes on the business’s liquidity, and build friction between customers and suppliers. A McKinsey survey found that global payments looks set to be a $2-trillion industry by 2020, accounting for a third of banking revenues. Nearly 70 percent of mid-sized B2Bs in the U.S. strongly preferred digital channels for payment approval and foreign exchange transactions emphasizing the dire need for efficient and real-time payments.
The payments industry is grappling for the backend
to catch up with the apparent convenience which payments apps seek to provide.
Over the years, the size of import and export trade between the U.S. and China
has burgeoned to more than half a trillion dollars.
Presently payment suppliers have to pre-fund accounts on either facet of dealing in native currencies. this may be costly and result in a poor deal for purchasers. Benefitting from favorable exchange rates which come as an added advantage of faster payments is crucial as well.
Blockchain is touted as the solution for global business ills by its ardent supporters. The next big disruptive technology in the payment space Blockchain can ease Cross-border payments in several ways. The distributed ledger technology (DLT) that underlies cryptocurrencies is currently being deployed by a few providers as the answer. Roughly seventy percent of prime international banks are experimenting with it, and recently some big implementations are underway
The distinctive point of blockchain – that reciprocally distrusting parties, geographically agnostic, will reach a sure agreement electronically is a banker’s dream. For about the same transaction fee and a more competitive forex rate, API-based blockchain platforms can significantly cut down processing time. This can significantly cut-down costs after nearly a century long-stagnant processing fee of traditional banking. Kuangyi Wei, head of research and market engagement at management consultancy Parker Fitzgerald estimates that it can save a third of the current operating costs.
Blockchain thus offers a cryptographically secure and trusty platform with high transparency. It further enables the bypassing of typical Forex rates. Small and mid-size firms like that of Mr.Chandra’s can thus reap the benefits of the best rates for their business.
Giants like IBM
are working on designing a universal, cross-country blockchain payments
solution that can herald instantaneously inter-bank transfer. This would
require streamlining multiple steps involving multiple stakeholders transacting
in different currencies.
Distributed ledgers are changeless databases area unit maintained by a network of computers, instead of a centralized authority, and secured by advanced cryptography. These are often described and clunky and difficult to maintain. They require enormous computing power to reconcile everything. The matter is further aggravated by the volume of cross-border payments, banks, and jurisdictions.
Centralized databases area unit still a lot of
economical than blockchains. Over-sold and over-hyped area unit different words
utilized in this context, managing expectations on what blockchain are able to
do are vital.
Operational resilience has additionally become a significant issue for central banks and regulators. considerations raised regarding DLT embrace privacy, security, measurability, and competition
Only time can tell if Blockchain will eventually become the panacea of payments it is heralded to be.
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


print('percentage_complex_words', percentage_complex_words(text))  # Output: 60.0

from textstat import textstat

fog_index = textstat.gunning_fog(text)

print('FOG index:', fog_index)

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

sentences = sent_tokenize(text)
total_words = 0
for sentence in sentences:
    words = word_tokenize(sentence)
    total_words += len(words)

avg_words_per_sentence = total_words / len(sentences)
print('avg_words_per_sentence:', avg_words_per_sentence)

import re


def count_complex_words(text, complex_words):
    # Split the text into words
    words = re.findall(r'\b\w+\b', text)

    # Count the number of complex words
    complex_word_count = sum(1 for word in words if word.lower() in complex_words)

    return complex_word_count


complex_words = ['analyze', 'sophisticated', 'articulate']

complex_word_count = count_complex_words(text, complex_words)
print('complex_word_count:', complex_word_count)

words = word_tokenize(text)
word_count = len(words)
print('word_count:', word_count)

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

print('avg_syllables_per_word:', avg_syllables_per_word)

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

