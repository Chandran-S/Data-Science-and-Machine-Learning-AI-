
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/difference-between-artificial-intelligence-machine-learning-statistics-and-data-mining/'
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
text = ('''What is the difference between Artificial Intelligence, Machine Learning, Statistics, and Data Mining?
“Data is the new oil” has become the most important trendline of the 21st century. The reason for this is the advancements in the fields related to data analysis. The field of AI Machine Learning Statistics and Data Mining all deal with data and are developing at such a staggering pace, that these fields have become the most popular buzzwords these days.
Buzzwords are originated through technical terms but often the underlying essence is ignored through fashionable use and mainly used to impress. This is the main reason for the misconception amongst people. AI, ML, Stats, Data Mining, and many other fields related to the analysis of data are most often mistook for one and the same thing, thus all these words are often used interchangeably to convey one and the same thing. But this is not true at all!
The only similarity between these disciplines is that all of these disciplines are related to the analysis of the data and converting this data into “information”.
In academics, while learning AI Machine Learning Statistics and Data Mining, the academic approach only wander in the technical definitions and concepts but the underlying essence and the aim of the discipline remain unexplored, same is the case with most of the articles out there which try to explain the difference. Thus, becoming the primary cause of confusion between learners. Hence, this article explains the difference by explaining the philosophy and the aims of each of the disciplines rather than wandering in the technical definitions.
The essential difference between these disciplines lies in their “aims” and the approach taken to achieve those aims.
The aim of Artificial intelligence is to understand intelligent entities. Then to satisfy this aim, we must first understand what is intelligence and what makes an agent intelligent agent? The answer to all these questions can be found by studying intelligent agents and the best example of an intelligent agent can be found by just standing in front of a mirror! Yes! Humans are the best examples of intelligent entities. Thus, in the 1900s, researchers began exploring the thought process, the reasoning process of humans as human beings were considered as an ideal intelligent agent. Mimicking human behavior became the aim of AI in the initial years of the research. After setting this goal, studies and experiments began and the most famous experiment conducted in the initial years to achieve this aim was the Turing Test! Turing defined intelligent behavior as the ability to achieve human-level performance in all cognitive tasks, sufficient to fool an interrogator. But this test received criticism as only mimicking human behavior is not exactly intelligence. Because intelligence should be related to the working of the human brain as without the human brain, intelligence has no meaning!
            Thus, mimicking human thought processes and reasoning became the transformed aim. The field of Psychology and philosophy also resonate with this aim that is to understand the human thought process. The difference is that AI not only tries to understand the thought process but to mimic it, build it. The collaboration of these fields resulted in models such as Neural Networks which try to mimic the function of neurons present in the human brain. So, basically, this initial aim was human-centered and humans were considered as the ideal intelligent agent.
Concurrently, the field of computer science was developing at a greater pace. With the advances in computer science, the experiments and theories could be easily tested and validated. As programs were being applied to solve real-life problems, it was found that computers performed better than humans at some tasks that are really complex for humans. One of the best examples of this could be the chess-playing program. An AI program defeated the world’s best chess player Garry Kasparov. This incident indicated that human intelligence is not the ultimate intelligence or else a human would have been able to defeat the AI program. This leads to a question, is human intelligence the ideal intelligence?
As computers became more advanced, they proved to be better than humans at certain complex tasks. That is why the new definition of intelligence was being related to the ability to solve cognitive tasks or problems. So, rather than considering the nature of agents, researchers began to study the nature of intelligence itself. Then the question comes how to test or validate intelligence? The best way to test intelligence is to solve cognitive problems. An agent can be said intelligent only if it can solve a complex problem. The problem-solving approach can be easily tested and validated on computers. Thus, some researchers began studying the ideal intelligence, and the selected agent to validate the experiments was the computer. So, a computer and problem-solving approach were adopted. So, the human-centered approach and computer, problem-solving approach are the two main aims of AI. Both of these fields have contributed to the field by giving valuable insights.
Both the aims are important and both of these collaboratively form the main aim of AI!
In the problem-solving approach, there is a big challenge that AI has to overcome in order to achieve its aim. Consider the example of solving a math problem. There are two cases by which intelligence can be tested in this problem-solving approach. Let us say two math problems are given for you to solve. The first problem is familiar to you and the second problem is not.
Consider the first problem. The first problem is familiar to you, that means you know how to solve such kind of problems as you have already solved some similar problems in the past. So, there comes a question, how our mind is able to solve that problem? The answer is, that you have solved similar problems in the past, thus you have learned from the past data, how to solve such problems, thus even if you haven’t seen that problem in the past, you will still be able to solve similar problems. This is one form of intelligence.
Consider another case where you are given a second problem where you have not solved such kind of problems in the past. Then to solve this problem, you will try to consciously gather and manipulate the given information so that you reach a certain conclusion. This kind of approach does not necessarily rely on the past data but completely on the reasoning process. This is the second kind of intelligence.
For AI to build intelligent agents, both of these kinds of intelligence must be developed in the agent. But, the reality is that AI has reached the point where it is able to build agents which can only learn from past data and find some useful information. AI today has not reached a point where it can build agents who can think on their own. That is the second type of intelligence.
So, the way AI is able to implement the first type of intelligence is through Machine Learning! So, the domain of AI which focuses solely on implementing the first kind of intelligence is in fact Machine Learning. That is the reason why ML is called the subset of AI! So, this is the main difference between AI and ML.
Technically speaking, “It is the field of study that gives computers the ability to learn from past data and find some meaningful conclusions, patterns without being explicitly programmed”. This statement needs some elaboration. The essence of ML is related to the process of “Generalization” and learning from past data. Generalization is an abstraction by which common properties of specific instances are formulated as general concepts or claims. Consider how we humans recognize daily life objects. If we see an animal, then we can easily recognize if it is a “dog” or a “cat”. It is a very trivial task for us but have you ever wondered how our mind is able to do it? The answer is Generalization!
If you were given a picture of a dog, you can easily recognize that it is a picture of a dog, because, our minds have abstracted the description of a dog and formulated it into a “concept” of what a dog is and these concepts became better and better as we learned from the past experiences of a dog. So, the way we think is dependent on the fact that things are represented as generalized concepts in our minds.
With generalization only, can come real “information”. So, we try to give computers the ability to generalize the “raw data” and convert it into “information” which can be patterns and trends in the data on their own.
This is the discipline that concerns the collection, organization, analysis, interpretation, and presentation of data. Statistics tries to deal with data with the only aim that is to explain it. So, it is the study of explaining the data itself! Statistics has two main domains which are Descriptive statistics and Inferential statistics.
Descriptive statistics deals with the explanation or description of data thus the name “Descriptive” statistics. It tries to explain as much information as possible, easily about the whole large data which would be a very complex task otherwise.
Inferential statistics try to make accurate inferences from the available small data. We use inference in many tasks in our daily lives. Consider a simple case of cooking a soup. After completing the recipe, you will taste a small sample, that is, a spoonful of the soup to check if the soup tastes good or bad. Depending upon the result of the sample, you make an inference about the whole soup that if the soup as a whole, good or bad. Similarly, in statistics, there are cases where you have to apply inference to have meaningful information. For example, consider a case where you cannot gather the whole data because it is very time-consuming and costly. In these cases, applying inference based on the available sample introduces uncertainty. That is where inferential statistics come for help.
So,the use of data in the context of uncertainty and decision-making in the face of uncertainty is what statistics deals with. So, however, and whatever the data, statistics tries to explain that data. This aim does not resonate with that of AI and ML, but statistics help these fields to correctly interpret the data!
“Data”, is not useful at all in its raw form. Consider examples of sensors used in industrial applications. These sensors might be used in a manufacturing plant to sense different properties like temperature, pressure, etc. The raw data generated by these sensors are not useful until and unless it is converted to a suitable form, then processed, analyzed to gather valuable insights, which can be used to solve a problem!
Due to its unique aim of capturing the essence of very large datasets, to gather insights, Data Mining is also referred to as “Knowledge Discovery”. That is why, Carly Fiorina, former CEO of Hewlett-Packard once said, “The goal is to turn data into information and information into Insight”. This statement completely explains the aim of Data Mining!
 So, the difference between AI Machine Learning Statistics and Data Mining lies in their aims. But the approaches taken in all of these fields, help in one way or the other in fulfilling the aims of the other fields. This is the beauty of these fields!
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

