
import requests
from bs4 import BeautifulSoup

# Example URL
url = "https://insights.blackcoffer.com/in-future-or-in-upcoming-years-humans-and-machines-are-going-to-work-together-in-every-field-of-work/"


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
text = ('''How humans and machines are evolving to work together?
In future or in upcoming years humans and machines are going to work together in every field of work. In upcoming days machines will be the need for every human being. Machines [AI technology] will do the work which humans are incapable of doing. Machines will partner and co-operate with humans.
According to the professor at the university of Washington, he explained that, as a result of AI, there will be more demand for existing jobs and new jobs will be created that are unimaginable today. Human workers and machines will work together flawlessly, complementing each other. Machines will learn to carry out easier tasks such as following processes or crunching data. They will also help the humans while difficult. Machines or AI will create a great job opportunities for humans in future. John Kelly ll, executive vice president of IBM once said that “Man and Machines working together always beat or make a better decision than a man or a machine independently.”
 In future, the three sectors of our country like agriculture sector, industrial sector and service sector are going to utilize the machines. So, that their work becomes not difficult. As of now, we can only see that for agriculture purposes various kinds of machines are used which we called as a modern farming method. Some major technologies [machines] that are harvest automation, autonomous tractors, seeding, and weeing and drones. As a result, farms can do agriculture peacefully. In the industrial sector also humans and machines are working together to increase production. Various types of machines are used in industries such as packing machines, loading machine etc. humans provide instructions to the machines and maintain the management in the company. Soon robots [machines] will assist doctors with surgeries. For instance, a doctor at remote location could direct a surgical robot to perform an open heart surgery. But the approaches option and decision will be left to experience and wisdom of the doctor not the robot.
What do you think of machines if they will make humans less or more in the field? Machines will push human professionals up the skillset ladder into uniquely human skills such as creativity, social abilities, empathy, and sense-making, which machines cannot automate. As a result, machines will make the workplace more, not less for humans. However, humans have to learn new skills throughout their lives. It is said that in the future 80% of process-oriented tasks will be done by machines. Quantitative reasoning tasks will be done approximately 50% by humans and 50% by machines, while humans will continue to do more than 80% of cross-functional reasoning tasks. According to Harvard research machines, algorithms can read diagnostic scans with 92% accuracy. Humans can do it with 96% accuracy. Together, it will be 99% accurate.
Human-machine collaboration enables companies to interact with employees and customers in the novel, more effective ways. Smart machines are helping humans to expand their abilities in three ways. They can amplify our cognitive strengths; interact with customers and employees to free us from higher-level tasks, and embody human skills to extend our physical capabilities. In the research, it was found that 1,500 companies achieve the most significant performance improvement when humans and machines work together. New machine systems have beyond-human cognitive abilities, which many of us fear could potentially dehumanize the future of work. Machines will indeed automate most repetitive and physical tasks, and part of quantitative tasks such as programming and even data science. According to D.E Shaw Group and professor at the University of Washington, explained that, as a result of machines, there will be more demand for existing jobs, and new jobs will be created that are unimaginable today. This is similar to how we couldn’t imagine a web app developer decades ago, and now millions make a living doing that today.
Machines are good at doing tasks with speed, precision, and accuracy. But machines are not very good at responding to unknown situations or making judgments. That part will be left to humans. Hence, the need for both humans and machines will be there in the future. Humans and machines have divergent skill sets that, when combined can transform the way we work. Machines have already infiltrated every aspect of our lives, and we must learn to live with them. In the future, human workers will interact more closely with humans.         
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

