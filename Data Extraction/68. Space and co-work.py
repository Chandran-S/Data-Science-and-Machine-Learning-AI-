
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/impact-of-covid-19-pandemic-on-office-space-and-co-working-industries/'

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
text = ('''Impact of COVID-19 pandemic on office space and co-working industries.
COVID 19 has bought the world to its knees. With businesses being shut, travel being banned, schools, and colleges being closed, we have observed an impeccable amount of sorrow and despise along with a great amount of mental torture and destruction. Though situations are improving now, and the lockdowns are being lifted, it is a fact that it cannot be shunned that this virus will have a long term impact on people, and the effects will not only be felt economically and physically but also mentally.
We’ve been reading articles regarding the impact of COVID 19 on the economy, education, mental health, and every other possible aspect. But one topic which has not been talked about much is the impact of COVID 19 on office space and co-working industries. Well, it is quite obvious that offices will start functioning now, but, the COVID period hasn’t ended. This means that social distancing will also have to be followed in offices; where it will be a problem to do so because of the lack of adequate infrastructure and space. The same will be the condition in co-working industries, wherein employees of different companies come and work together under the same roof.
However, these problems are quite obvious (though people haven’t thought about them much), but the fact which people fail to notice and understand is that the meaning of space is not only limited to infrastructural space which is measured in square foot or square meters; but it also extends to metal space, or for that fact, space in the minds of people for others and their thoughts as well as opinions.
Coronavirus has left a big scratch in our country; just the way a car needs repair after an accident, our country will also need repair after the control of this virus. But, unlike a car, our country will require a large amount of time to get restored and to stand straight without any support.
Needless to say, the economic stability of our country will definitely be hit because of this reason. But the problem is since the earnings of the employers will reduce, it will subsequently lead to a drop in the salary of the employees. While some employees might be preferred by their bosses and may receive a raise regularly, others might be subjected to unfair treatment and made to do the same amount of work for a lower salary.
People tend to overlook the thin line which exists between cooperation and competition. The problem doesn’t arise when such a situation happens wherein an employee is getting paid more than the other, but instead, it arises when people start to compare their earnings as well as benefits to that of others. And in today’s world, where people keep money at the top of their priority list, comparisons will play a very important role in the performance of employees and their company.
Since there will exist inequalities and partialities in companies, there will also exist a sense of hatred and competition among the employees of the same company; who will compete to get a better position or a raise in their salary. This may seem beneficial to some as it might compel the employees to work harder and better, but however, will bring huge losses in the long run. The fact is; employees might end up doing unethical acts which will help them to make a better impression in their boss’s eyes.
Moreover, since there will be comparisons among the employees, it might happen that the ‘Employee of the Month’ ends up being hated by all other employees in the company. Another situation that might arise is that of bluffing and pretending; wherein employees tend to show that they work very hard but they don’t. This might be practiced by showing the boss that they are working when he is on around, and then chilling the rest of the time when the boss goes home or sits in his office. In fact, employees might also lose interest in working if they sense repeated partialities and inequalities; which they might show by quitting the jobs or not performing to their best.
Such situations will definitely arise in companies and offices; wherein employees will compete for attention and resources; which will have a great impact on the company and its functioning. And all this will happen because of only one thing; lack of office mental space i.e. space in the minds of people working together in a particular office for cooperation and teamwork, along with ignorance of the fact that things will get better and eventually the same after some time.
Another kind of industry and companies that will be affected due to the long-lasting impact of COVID 19 is the co-working industry and offices. This is because, there will be a shortage of resources and the income of people, but the maintenance cost to be born for the infrastructure, as well the rent will remain the same. Because of this, employees will be told by the company owners to exploit the resources present in the offices and use them to the fullest; to compensate for the losses being born and to make better use of the money being used. However, the number of resources will be the same, and so, there will be a case of competition between employees of different companies to use resources and take benefits of the same.
If the above two situations happen to occur at the same time (which they most likely will), we will observe a scene of intra-company as well as inter-company competition. Also, our personal relations with other employees in the company might also be affected due to this comparison and competition; which itself will be a result of limited mental space.
COVID 19 will have severe impacts on the office (mental) space of companies and people.  People’s thoughts, how they treat others, along with accepting defeat and rejection will be an important aspect of everyone’s lives. Moreover, we might also observe a shoot in the intensity and frequency of office people going to psychologists; because of repeated rejection and unfair treatment. However, we need to cope with this thing and realize that it is indeed very important for us to start preparing to expand our mental space and that this competition and comparison will not stay for long and should not affect our personal relations and life at any cost.
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

