
import requests
from bs4 import BeautifulSoup

# Example URL
url = "https://insights.blackcoffer.com/what-jobs-will-robots-take-from-humans-in-the-future/"

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
text = ('''What Jobs Will Robots Take From Humans in The Future?
Introduction
AI is rapidly evolving in the employment sector, particularly in matters involving business and finance. Finance, management, economics, and accounting are now among the most popular university courses globally; particularly at the graduate level, due to their high employability. However, the evolution of machinery in industries is changing that. According to research, 230,000 jobs in these sectors may be replaced by AI agents in the next 5 years. This is due to the nature of the work, as employees are responsible for tasks such as data analysis and keeping track of numerical information; which machines excel at. Large, complicated data sets can be analyzed faster and more efficiently by AI-powered computers than by people. Algorithmic trading procedures that produce automated deals are less likely to be produced without minute errors when undertaken by humans. In such matters involving industrial work, a subsection of artificial intelligence is used; namely, machine learning.
Machine learning is a term used for the application of artificial intelligence (AI) in systems, which involves them assimilating and processing information by gaining experience. It is mainly concerned with the development of technology and computer programs.
 Its improvement process is mainly divided into seven steps; which start with the collection of data. This data can be collected from an internal database or perhaps an IoT structure. The second and most time-consuming step is cleaning, preparing, and rearranging the data; which involves the recognition of outliers, trends, and missing information so that the outcome is as accurate as possible. The third step consists of formatting data; which is useful if you source your information from a variety of sources. Step four is where AI comes into place. Self-service data processing tools may be useful if they provide intelligent services for matching data attributes from distinct databases and intelligently integrating them. The data is then arranged to better represent a specific pattern. Lastly, the data is divided into two sets: one for training the algorithm and one for analysis.
There are three types of machine learning; supervised, unsupervised, and reinforcement learning. The most common of the bunch is supervised machine learning; which is based on accurately labeled data. The machine is given a collection of information, including the outcome of the operation. The rest of the information is referred to as ‘input features’, which ‘supervises’ the machine by guiding it to establish the connections between the variants.
In the case of unsupervised learning, the machine isn’t given labeled sets of data to be divided into, allowing it to recognize and create its own patterns within the information provided. Since computers have the capabilities of identifying distinguished similarities, this method helps with the classifying of such data.
Reinforcement learning comprises experience-based learning. Similar to people, they learn due to a reward and punishment system based on their actions.
These variations of system learning have recently been integrated into business and finance, which is elaborated on in the following passages.
Machine learning in finance and banking
There are many ways in which machine learning is used in finance and banking. The most common place it is used is to detect any frauds. Fraud is the most common problem in financial service companies and it accounts for billions of dollars in misfortune each year. The most common frauds are credit or debit card usage by a stranger, document forgery, and mortgage fraud.
Usually, finance companies keep an enormous amount of their data stored online, and it increases the risk of information being accessed without authorization. With expanding innovative headway, misrepresentation in the monetary business is currently viewed as a high danger to significant information. Fraud recognition frameworks in the past were planned dependent on a bunch of rules, which could be effortlessly be bypassed by current fraudsters. Therefore, most companies today leverage machine learning to combat fraudulent financial transactions.
 Machine learning works by looking over enormous informational indexes to distinguish unique or suspicious activities for additional examination by security groups. It works by looking at an exchange against other information focuses – like the client’s record history, IP address, area, and so forth.  Depending on the type of exchange taking place, the program can automatically refuse a withdrawal or purchase until the person makes a decision.
Examples of fraud detection software used by banks include Feedzai, Data visor, and Teradata
Machine learning is also used in algorithmic trading, portfolio management, loan underwriting, chatbots, and improving customer service.
Algorithmic trading is an interaction for executing orders using computerized and pre-modified exchanging guidelines to represent factors like value, timing, and volume. An algorithm is a set of directions for solving a problem. Computer calculations send little parts of the full request to the market over the long run.
In contrast to human dealers, algorithmic exchanging can investigate enormous volumes of information every day and therefore make thousands of trades every day. Machine learning settles on quick exchanging choices, which gives human traders a benefit over the market normal. Likewise, algorithmic exchanging doesn’t settle on exchanging choices dependent on feelings, which is a typical constraint among human dealers whose judgment might be influenced by feelings or individual desires. The exchanging technique is generally utilized by multifaceted investment administrators and monetary foundations to automate trading activities.
In the banking industry, organizations access a large number of shopper information, with which machine learning can be prepared to work in order to simplify the underwriting process. Machine learning calculations can settle on speedy choices on endorsing and credit scoring, and save organizations both time and monetary assets that are utilized by people.
Data scientists can train algorithms on how to analyze millions of consumer data to coordinate with information records, search for interesting special cases, and settle on a choice on whether a shopper meets all requirements for an advance or protection. For instance, the calculation can be prepared on the most proficient method to examine shopper information, like age, pay, occupation, and the buyer’s credit conduct.
The benefits of machine learning
As with any form of revolutionary technology, the usage of machine learning has been debated over as its beneficial properties have been weighed against the possible disadvantages. It’s been observed that upon correct usage, it may be used to solve a wide range of business challenges and anticipate complicated customer behavior. We’ve also seen several of the largest IT conglomerates, such as Google, Amazon, and Microsoft, introduce Cloud Machine Learning platforms.
In terms of business, ML can aid businesses by identifying consumer’s demands and formulate a pattern based on them; allowing companies to reach out to such consumers and maximize their sales. It eliminates the need for expenses on some maintenance by reducing the risks associated with unexpected breakdowns and minimizes wasteful costs to the firms.  Historical data, a process visualization tool, a flexible analytical environment, and a feedback loop may all be used to build an ML architecture. The input of data is another agitating chore for businesses in the present, which is where machine learning can step in for manual labor. This eradicates the possibility of errors caused by manual labor causing disruptions and provides employees with extra time to handle other tasks.
In the financial sector, machine learning is already utilized for portfolio management, algorithmic trading, loan underwriting, and fraud detection. Chatbots and other conversational interfaces for security, customer support, and sentiment analysis will be among the future uses of ML in banking. Another aspect of business involving artificial intelligence includes image recognition, which entails a system or program that recognizes objects, people, places, and movements in photos. It identifies photographs using an imaging system and machine recognition tech with artificial intelligence and programmed algorithms.
The downfalls of machine learning
With all the benefits of its advancement, machine learning isn’t the most perfect thing. There are several disadvantages which are information acquisition, time and resources and high errors, and wrong interpretations. One of the major hurdles is the amount of finance needed to invest in machine learning for it to be a successful project. More issues have to do with the fact that AI requires gigantic informational indexes to train on, and these ought to be unbiased, and of good quality. There can likewise be times where they have to wait for that new information to be produced. Machine learning needs sufficient opportunity to do the calculations to learn and adequately to satisfy their accuracy and relevancy. It also needs huge resources to work. This can mean extra requirements for the computer to work. Another significant problem is the capacity to precisely decipher the results produced by the calculations. You should likewise cautiously pick the algorithms to get the wanted results.
Conclusion
There have been various reports in the past and current years which claim that a significant piece of the human labor force will be replaced via robots and machines in the years to come. With excessive innovative work being led in the field of computerized reasoning, many dread that a significant job crisis will unfurl since numerous positions are all the more precisely and productively performed with the use of machines. In countries like Japan, mainly computer programs and AI is used in the secondary and tertiary sectors. From cleaning the house to depositing money in banks, everything is done by AI. However, AI cannot replace humans in the future. Humans have several capabilities which, even after several technological advancements a machine would not be able to have. These capabilities include creative thinking and creative problem solving, and human connection. For example, when a child goes to a doctor to get an injection, a nurse always relaxes the child to not be afraid of the needle. A machine’s touch would not be able to soothe a child. Another example could be how humans tend to share things with each other and be open about it, a machine will not be able to do so since it is only programmed to things it has been told to do. Like computers, AI will not replace us but would however complement us to make daily work easier and less time-consuming. Without humans themselves, there is no future for AI.''')
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
