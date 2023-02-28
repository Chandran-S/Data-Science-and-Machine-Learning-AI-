
import requests
from bs4 import BeautifulSoup

# Example URL
url = "https://insights.blackcoffer.com/how-machine-learning-will-affect-your-business/"


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
text = ('''How machine learning will affect your business?
Machine learning techniques may have been used for years, but recently there has been an explosion in their applications. In fact, in a recent Q3 earnings call, Google CEO Sundar Pichai said “Machine learning is a core, transformative way by which we’re re-thinking how we’re doing everything.” And they’re far from the only business making that claim.
In the past, successful use of machine learning algorithms required bespoke algorithms and huge R&D budgets, but all that is changing. IBM Watson, Microsoft Azure, Amazon, and Alibaba all launched turnkey cloud-based machine-learning SaaS solutions in 2015. At the same time startups like Idibon, MetaMind, Dato, and MonkeyLearn have built machine learning products that companies can take advantage of.
Gartner already puts machine learning at the top of its hype curve, and no: machine learning won’t replace all of your employees with computers or suddenly double your revenue. But that doesn’t mean that it can’t give every business a competitive advantage. There are plenty of business processes that can significantly benefit from machine learning. So how does machine learning change the way businesses operate?
First thing’s first: Machine learning needs training data and training data costs money. Especially training data labeled by humans. Let me explain. To make machine learning work for business, the algorithm needs to see lots and lots of examples of what it’s supposed to be doing. If you want an algorithm to tell you if a sales lead is good, you need to show it lots and lots of examples of good sales leads and bad sales leads. If you want an algorithm to tag the support tickets you need to show it many examples of support tickets. If you localize your algorithm to a new language you probably need to collect lots of examples in that language. In some instances, a company may have those training sets in-house. For example, a bunch of disqualified or qualified leads. But say you haven’t labeled each of your support tickets as they’ve come in over the year. You’d need to have people either in-house or en masse via a data enrichment platform -label those tickets. The machine will then look at those judgments and start finding connections and patterns it can learn from.
Machine learning is much cheaper and more efficient than people when it works well. The downside is that it often works well in 80 percent of the cases and badly in 20 percent of the cases, and lowering the 20 percent error rate is hard, if not impossible. But even an 80 percent accurate algorithm can save you a lot of money because good machine learning algorithms know where they are accurate and where they are more likely to have errors. Smart companies take the cases where the algorithm has high confidence and uses those directly while sending low confidence cases to humans. Banks have been doing this for years. When you put a check in an ATM, an algorithm tries to decipher the numbers on the check. If you have really sloppy handwriting or the ink is smudged the algorithm passes the task to a human. This design pattern saves banks lots of money while preserving a very high level of accuracy.
A huge benefit of machine learning is that it can turn part of your variable cost into more of a fixed cost. If you use humans to handle cases where that algorithm is struggling, you are creating the perfect training data to feed into your algorithm. This is a well-studied technique called active learning it turns out that training data labels collected on cases where the algorithm has low confidence help the algorithm learn much, much more efficiently.
As the algorithm becomes increasingly more accurate, the unit economics of your business process become better and as machine learning becomes able to handle more cases, the expensive humans are only called in on the toughest, rarest situations. That means you use the best of both human and machine intelligence in tandem: leveraging the speed and reliability of computers for the easy judgments and the fluency and expertise of humans for the difficult ones. And if that sounds like smart business, it’s because it is.
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

