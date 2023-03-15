
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/how-voice-search-makes-your-business-a-successful-business/'


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
text = ('''How Voice search makes your business a successful business.
Finding ways to make using the Internet easier is something most modern consumers are passionate about. Since the rise of AI-infused technology like Amazon’s Alexa or Apple’s Siri, voice search has grown increasingly popular. In fact, nearly 60 percent of people all over the world use voice search at least once a day.
As this technology grows more popular and complex with each passing day, business owners are starting to take notice of voice search. Capitalizing on this trend is only possible when optimizing the content on your website for voice search. Are you trying to make your website more voice search-friendly? If so, check out the following tips.
In the past, if you wanted relevant answers from a voice recognition technology, well—good luck with that. But today, machine learning systems can compete with people in terms of accuracy. Google’s voice recognition, Google is now able to understand human language with 95 percent accuracy. These improvements mean that, while you can trust good voice systems to match customers with the right products, services, or information with increasing degrees of nuance, leading businesses already have set customer expectations for delivery. Thanks to machine learning algorithms that can detect speech and respond with meaningful results.
Because voice search systems are at a point where they actually can perform reliably and meet customer expectations just as well (if not better) than traditional query options, customer trust in them is growing:
Yet, only 4% of businesses are properly optimized for voice search. What’s with the disconnect? In other words, business leaders are at an ideal intersection where reliable systems are available and there are still many customers who haven’t been reached or who might want more out of the systems they currently use. Bringing those systems and customers together will help companies avoid being left behind.
When people use voice search, they don’t just want to locate a great pair of shoes or a TV. People want all kinds of other information, such as your store hours, how to connect to support specialists, and when you’re having your next sale. This is partly why some experts have predicted a “totally different internet” within the next five to 10 years, one where voice-activated chatbots have all but replaced the e-commerce channels we’re used to using. 
Shoppers are also after general tips that can take some of the friction out of everyday life—think life hacks and how-tos. In fact, words like “how,” “what,” “best,” and “easy” are among the top voice search queries. This means that when it comes to online marketing, you probably need to change your entire optimization approach, taking elements like grammar and semantics, the structure of your site, and structured data markup that influences Google’s ability to find your content. Of course, optimizing from the start, not as an afterthought, is ideal.
Voice Search can be used on both desktop and mobile searches. We can see that customers want different types of information from voice systems, they also can use it at many points in the customer journey. For example, about half of shoppers use voice to research products, even “near me” searches results are mostly done by customers. Customers also use voice to:
These statistics show you should think beyond just having customers find you or your products. Voice search and commands can take your buyers from start to finish, so give your customers as much convenience and satisfaction as possible by integrating voice options into more types of interactions.
A survey showed that, once a consumer makes a local voice search, their next most common action is to call the business (28%). Customers also are highly likely to visit company websites (27%), show up at the company’s location (19%), and do more research into that business or additional businesses (14% and 12%, respectively). So, in short, while voice services can allow a customer to complete many steps without human interaction, you shouldn’t see it as a total substitute. Many people are going to take further steps after the initial search, and they still will want to hear a human voice respond back once in a while. They are going to have more questions, and they’re willing to physically connect with you and what you offer. Don’t drop the ball in other areas, like making your site aesthetically pleasing and easy to navigate, having enough staff ready to chat, listing an accurate contact number, or keeping your store well-stocked. Voice search already is shifting the way customers engage with brands, but there’s still time for companies of any size to get involved with voice systems in ways that can benefit both the customer and the business’s bottom line. The next step is to find voice search solutions. However, as with other technologies, these are not necessarily one size fits all, meaning it’s critical to shop around and be specific about your goals. If you can customize and update your options in a scalable way, they’ll be even more effective for your business.
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

