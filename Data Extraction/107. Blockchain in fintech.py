import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/blockchain-in-fintech/'



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
text = ('''Blockchain in Fintech

Blockchain or the Distributed Ledger Technology has
already been around for some time now. Cryptocurrencies like Bitcoin, based on
the Blockchain technology, are making entry into the formal finance world and
increasingly becoming a medium for exchange. Though there are many potential
use cases for Cryptocurrencies and the Indian Government is very critical about
implementing the technology in banking and finance, it still has a lot of
issues to be resolved before we can commercially use it for transactions. 
Some of the most important drawbacks of
Cryptocurrencies like Bitcoin right now are as below:
So, it becomes important to have a cheaply available
validation mechanism that can be carried out using a basic broadband connection
in-home/small business computers. If two people have personal transaction
history, it’s unimportant that it should be posted on the main blockchain every
time a transaction takes place as it could bloat up the blockchain. Moreover,
everyday transactions using cryptocurrency is not possible as the cost of
posting a transaction for. Let’s say buying a coffee, can be more than the
price of coffee itself.
Lightning Network is the latest disruptive technology
being researched currently in Fintech to make cryptocurrencies scalable. It
basically adds
an extra layer on top of the main Bitcoin Blockchain and enables payment
channels between two parties over this layer. Micropayments for everyday
transactions can be very easily carried out using this channel. Here is how it
works:
Suppose A buys a coffee
from the nearby coffee shop B daily on her way to work. A and B decide to open
a channel on the Blockchain for their everyday transactions. Once the channel
(or the Lightning Network) is open, the two parties can keep a track of their
daily transactions on this ledger. At the end of the month, with the ledger
reflecting balances left with each party, they decide to close the channel.
This final transaction is then recorded on the main Bitcoin Blockchain. This
reduces the cost of the transaction and enables micropayments for everyday
transactions.  Micropayment channels can defer updating the main
blockchain with remaining balances at a later date. A large network of
micropayments can be created as a secondary layer on top of the main blockchain
to address the scalability issue and reduce transaction fees. Although this solution
is very much in the initial stages, resolving the scalability issue and
reducing transaction fees by taking the transaction out of the main blockchain
will give a big boost to the use of Cryptocurrencies.
The two parties exchange a single key for validation
of their spend transactions. They can conduct an unlimited number of transactions
on this channel and finally when one party closes the channel, the balances
will be added to the blockchain.
The network is globally scalable since to reach any
person through connected channels network, you just need to know the path to
connect to that person through leveraging your existing network. Once the
network develops in future, the requirement to open a separate channel too
would cease to exist since the algorithm will automatically find the shortest
route to connect to the person whom you want to pay based on the exhaustive
peer to peer network. This would ease inter-country transactions and money can
be sent across internationally within seconds.
Since the channel for Lightning Network is separate and not connected with the main Blockchain, it will not be backed by the security of the original Blockchain. This can be a major concern for its widespread adoption and the companies should research extensively into this. Another challenge is its dependency on the internet. The Lightning Network is not capable of storing coins digitally in a hardware wallet. This makes it prone to hackers since the wallet will be available online. Also, transferring large amounts of money is something that needs further detailed research, since bigger payment might struggle with routing in the network.
Nevertheless, this technology is really fascinating and has a very broad usage through the implementation of Bitcoins and Distributed Ledger in everyday life. Once this is rolled out for customer experience, micropayments will be possible between two parties on Blockchain, just as we currently have wallets to transfer money for small purchases. The Lightning Network has a long way to go and is surely going to become the next big disruption in Fintech domain after Blockchain.
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

