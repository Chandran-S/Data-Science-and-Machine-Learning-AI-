
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/why-does-your-business-need-a-chatbot/'

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
text = ('''Why does your business need a chatbot?
Mr. Sakamoto is a bonsai artist, lives downtown Kyoto, Japan. Bonsai is a Japanese art form that is the cultivation of small trees in a small or medium container. He is in this field of business for the last ten years and he loves what he does on a daily basis. People from different cities in Japan, come to buy his artwork. But, in the early stage of his business, he was just another regular Bosai artist. His popularity has increased because of the stories he tells for every art he is selling. His customer immediately loved him for this while buying A Bonsai art. He loved to sell his customer with his artwork and a story related to that product which resurrects his presence through his art in customers. Soon enough his business thrived and nowadays he is always busy. After the digital revolution, although his sales increased, he was not happy. He was not satisfied. Something is eating up his mind day by day. He could not understand what’s the reason behind his remorse.
A chatbot is an awesome invention of the digital revolution. It is the real-life embodiment of Customer Engagement. Every business needs this feature- will be an understatement. Because Customer engagement is the base of any business. The chatbot does the exact thing digitally. It helps customers to answer the queries and tell about the product that the business has to offer is a fully automated way with vibrant customizations.
Mr. Sakamoto is running his business for the last fifteen years now. And now he is happy that he was not for the past couple of years. The rationale behind his happiness is the chatbot. Though he has a website and app to sell his Bonsai art, he could not engage with the customers in the way he wanted to. He has trained his employees and apprentices, but they are not providing the same service in the way he does. He was unsatisfied and the chatbot relives him from his sadness. He has done all that at affordable prices in no time.
Let’s understand the Mr. Sakamoto’s “Mondai Nai”- perspective about why chatbot is the answer of his problems.
This is the fundamental and root cause for which Mr. Sakamoto has to implement the chatbot in the first place. His stories are embedded with each art. The product which he sells needs this story to be with it. His stories which were not been told to all the customers before are now sharing with this chatbot and building the trust between them. We all can agree on this one statement that trust is the most impactful aspect, maybe the best of aspects, in the relation between any customer and business-owner, whether the business is small, medium, or large.
Millennials, a word that was not introduced before 1990. The generation which is stuck in between Gen-x and Gen-y. A generation of people accustomed to both digitization and old culture. They are the most targeted audience in this era. What else would be more approachable to this generation than a chatbot? It’s easy, less time consuming to use. Mr. Sakamoto is now one of the few Gen-x persons who can do business with the millennial so easily.
Mr. Sakamoto has some employees and apprentices. He has trained them as per his needs. But there is a limit. They can’t deliver the exact things that he can or the way he does. Neither chatbot can do that exactly, it will accurately follow the instructions. Not only the chatbot solve the queries of the customer in his way, but also give a recommendation to the product in the way he wants too. Is there any more attribute he would care about that? Absolutely no!
The most expensive thing for any businessman is the return of their customer with an empty hand because of his unavailability. A chatbot is a perfect way to solve it. For Mr. Sakamoto’s customized chatbot is not only present for customers in their availability, but also It was giving the response to the customer in all the time zone, all over the world. Mr. Sakamoto wouldn’t be happier for this feature.
In the era of globalization, what is more, helpful than a chatbot? It can be customized in any language. It can be accessed by any human being residing in any part of the world. And at the end of the day, when that foreigner is happy with the buy and the felicitation and says “Arigato” but in his own language, at that moment he feels so warm and happy.
Handling a lot of customers is really hard. He assigned employees, but the rate is growing fast. So, in the competition of the market, Mr. Sakamoto took the help of a chatbot. Now this automated customized application is handling a greater number of customers than his employees can do. His business is growing exponentially.
Now, when you found out why Mr. Sakamoto is happy and his business is growing in this competitive industry then, do you still have a question in mind that why does your business need a chatbot?
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

