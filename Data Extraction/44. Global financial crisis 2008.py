
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/global-financial-crisis-2008-causes-effects-and-its-solution/'


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
text = ('''Global financial crisis 2008 causes/effects and its solution
The Financial Crisis of 2008 started as a crisis in the subprime mortgage market (i.e. in a market where lending of loans is done to people who may have difficulty in maintaining the repayment schedule or in simple terms, loans were given out to people without proper checks and low credit scores) which ultimately led onto a huge global collapse.
The Financial Crisis of 2008 or Great Recession is considered the worst economic crisis since the Great Depression.
In the year 1996, there was a dot-com boom (or otherwise known as the dot com bubble) in the United States, a period of massive growth in the use of the internet because of which the stock market prices started increasing rapidly. However, around the year 2000, it dropped which led people and investors to withdraw their investments from the stock market rapidly. It led to the decline of the price of shares in stock markets and the interest rate plummeted to around 1% very quickly in a short span.
Investors were now looking for a brighter option than investing in stock markets.
 Fig – Rise and fall of the Dot Com Bubble
As interest rates went lower and lower, real estate prices started rising and the US Govt also encouraged people to buy houses and properties. The demand for the same started rising rapidly and investors now found a great option to invest in (i.e. in Real Estate).
During that same time, Investment Banks saw an opportunity, chimed in and started buying loans from Banks in bulk and clubbed multiple loans under a complex derivative called CDO (Collateralized Debt Obligations) and started providing it to these Investors after getting a credit rating of “AAA” (Very Safe Investment) from the Credit Rating Agencies. Investors naturally fell for it and started buying these CDOs.
So now, the risk factor of these loans got transferred from the Banks to the Investment Banks and then again to the CDO Investors.
With the high buying demand of CDOs, Investment banks started demanding or pressurizing Banks to provide even more Loans so that they can provide more CDOs to the Investors. However, Banks had already provided loans to people with good credit history and regular income people. But in the hope of getting even more credit from Investment Banks, these same banks then started giving out subprime housing loans to people with low credit scores.Approximately $174Bn worth of loans were given out during the period 2000-2007 and most of them were clubbed as CDOs with a “AAA” rating from the Credit Rating Agencies. Approximately 70% of these CDO’s were marked with a “AAA” rating.
Investment Banks and Credit Rating Agencies were now enjoying large profits during this time. Moody’s (a credit rating agency) profits increased 4x times during that period (2000-2007).
Looking at the huge profits being made by the Investment Banks and Credit Rating Agencies, Insurance companies (like AIG) now started giving out insurance on these CDOs to the investors and they called it CDS (Credit Default Swap). AIG believed that since the CDOs were rated as “AAA” (Very Safe Investment), the failure chances of these CDOs were very minimal. They misjudged or were unaware of the fact that some of the loans that were clubbed under these CDOs were Sub Prime loans.
Now CDO Investors started buying out CDS from AIG and other companies to safeguard and protect them from any losses. AIG then started making huge profits because of the premiums that the investors had to pay. But they never realized the outcome if by any chance the CDOs fail at some point in time. Thus, the risk factor again got transferred from the CDO investors to the Insurance Companies.
Fig – Flow Diagram showing how the Risk Factors got transferred
Coming to the loan borrowers now, Sub Prime Borrowers from banks were unaware of the fact of Adjustable Rate Loans (interest rate of these loans keeps changing) and thus had to pay lower interests at the start but more interest rates later on.
The borrowers started defaulting on these loans when the interest rates increased dramatically around 2007 and thus banks had to then resell those houses to make up for the loans defaulted.  Added to the problem was the fact that borrowers were not spending any amount of money from their pockets while taking loans and banks were providing the full amount of loans. Almost 50% of the borrowers did not pay anything from their own pocket and bought the home only using the housing loan.
This led to a huge increase in the defaulters of borrowers and ultimately banks had to auction houses to gain credit back. With the high-interest rates and no one to buy the auctioned houses, this ultimately caused a chain reaction and banks were no longer receiving credit. The prices of Real Estate started falling drastically and people with good credit scores who had earlier taken housing loans also started defaulting because the price of their houses/homes fell below the loan amount that they had taken earlier. Banks stopped receiving money and also because of this chain reaction, the value of CDOs ultimately came down to 0.
Some Investors went for huge losses and few Investment companies went bankrupt (ex – Lehman Brothers). Moreover, insurance companies had to pay back those investors who had taken insurance. As a result, some of the insurance companies also went bankrupt. AIG too almost lost about $100Bn in paying back the investors who had earlier insured their CDOs. However, the US Govt. finally decided to bail out AIG in order to save them from going bankrupt.
CDOs and CDSs were not regulated during that time by the Federal Reserve and thus this whole situation ultimately led to credit crunch (became very difficult to get loans) and the whole economy of the US underwent a crisis that led to a global impact all around the world. Unemployment increased manifold and many new businesses had to shut down. Global trades all around the world also saw a crisis and finally, Global Recession hit the World.
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

