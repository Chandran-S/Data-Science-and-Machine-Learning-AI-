import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/business-analytics-in-the-healthcare-industry/'


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
text = ('''Business Analytics In The Healthcare Industry
Analytics is a statistical scientific process of discovering and presenting the meaningful patterns that can be found in data. While business analytics refers to the skills, technologies, applications, practices, computer programming and operations research for the continuous exploration of data to gain insight that drives business decisions.
Business analytics technology is helping healthcare organizations regulate existing data to improve clinical and business operations. In addition to this, analytics help to identify the symptoms of diseases in preliminary stages itself along with suggesting possible remedies for the same without any need of the intervention of a human being. With the help of analytics and chatbots patient can now look for expert doctor advice in the comfort of their home free of charge in most of the cases with as easy accessibility as one-touch from your smartphones.
According to
the survey conducted by Health Catalyst, a whopping 90% of the respondents
admitted that analytics is going to be either “extremely important” or “very
important” to their organization within the next few years. And the respondents
also rated the importance of healthcare trends and the role played by analytics
in them.
Analytics is thus becoming very crucial in tracking different types of healthcare trends. Advanced analytics touches every aspect of healthcare software systems including clinical, operational and financial sectors.
Healthcare organizations have begun to adopt technologies like PACS imaging systems and EMRs or Electronic Health Records that attempts to make sense of the massive data that flows through the system (both structured and unstructured). Hence, it is important to know what are the tools that can extract information from the data to generate value and enjoy operational, financial and clinical insights.
There are genome analyzers and other analytics tools in the market that help in understanding the facts, and eliminate unwanted/useless details to extract only what’s needed. The end result of this is better clinical outcomes for the patient. There are several ways in which healthcare organizations can make use of information collected through various sources.
The following examples indicate successful applications,
built on a foundation of advanced analytical capabilities, in the healthcare
industry:
Predictive analytics is key to enabling hospitals to
properly manage their readmission rates and sidestep costly penalties while
simultaneously addressing important aspects of patient treatment and care.
Analytics are changing roles in the healthcare industry.  An increasing number of informed patients are taking more responsibility for their own care.  Likewise, physicians are finding more satisfaction with their positions as positive effects increase.  More time spent with individual patients has increased which gives physicians the chance to form a trusted relationship with the patient.  Physicians want to spend time with their patients – to know them, interact with them, and help them.  When the time to develop a relationship is diminished, the physician is less satisfied with his or her profession.
Analytics has
changed the way the healthcare world operates.  With the ability to
transform the way medicine has been practiced for years, analytics have
resulted in improved health, reduction in diseases, and more satisfied patients
and physicians.
Combine artificial intelligence with data analysis and machine learning IoT, and it is easy to provide proactive care to patients. Hence, it would be a good move to invest in analytical solutions that can control and mitigate clinical and financial risks, with new payment bundles and models to go with it.
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

