
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/impact-of-covid-19-on-the-global-economy/'


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
text = ('''IMPACT OF COVID-19 ON THE GLOBAL ECONOMY
It was 14th March and I was
altogether prepared for my last theater competition of this very year. 
I only knew in my core how much this competition meant for me. In the morning I made some last-minute calls to my diligent team who were ceaselessly practicing for the past few days to confirm to me that they were all coming for the competition because of the alarming Coronavirus ball game going outside the market. It was not like I wasn’t aware of the situation but I believed that I would be able to handle it because there’s a quote in theatre that says, “Fear is only as deep as the mind allows” and to some length I was fearless. My passion, fondness, and attachment for the theatre were above all the notices and announcements that we’re rolling out.
But then at once, I received a call from Mr. Sandeep, the head of the event who told me that the event is put off till the next few days and from that precise moment onwards the journey of hard and strenuous days began and it in point of fact felt like it was all meant to happen. The competition was postponed, the team was stuck in their homes grudgingly, time was haunting us all because of this Coronavirus. These were some of the cardinal ways in which this virus affected us but there was more to this sickness. This very sickness acted like a sadist. I was impressed as I found someone who could act better than me. It has reversed the ways we used to work, play and exercise and the world solution that was asked to practice by the professionals were Social Distancing. 
This virus was wild and dangerous towards the people who are in declining health conditions but obviously it could not take off his hands from the young people as well. The Coronavirus cases and deaths were heightening to a great extent. Also the major problem that came in the light was people being fully unprepared and this led to the public becoming snappier, intolerant and non-supportive needless to say as this virus was acting like a sadist. But by all means, our government knows how to best handle the affairs, they decided to shut down schools, colleges, bars anything that would involve the gathering of more than 10 people. With disinformation coming from all around related to this infection it becomes hard to judge for the audience what is true and what is false and it worsens when the public consumes this information by working from home. This virus was not only hollowing people from deep inside but was also supporting and increasing some other country-related issues one of which is Fake news which is a serious problem.
With fake news traveling around the world, people come and start to blabber. People start offering numerous solutions and want that most of the people should follow it. They start acting like saints who know all the solutions to this pandemic. Some of the solutions are convenient to follow but some are not and are unhealthy which creates immense confusion and arguments. But the best way to stop the confusion and to avoid these people is to avoid their arguments and to not convince them with your points because it is rightly said a man convinced against his will is of the same opinions still. The thing to do is to look after your family and to avoid shady and deceitful people.
But one thing that can’t be avoided is our expanding and maturing industries that lend a hand in developing our nation but no power to choose is left with our industries but to be at a standstill in this current moment as Coronavirus right now is winning the game. The untold number of industries is growing down and a major one is our Tourism industries and if this continues to happen then in no time there will be a loss to jobs close to more than 75 million. Many working people like in the tourism industry and not only in this peculiar industry but also in various others, people are getting suspended because of a hefty amount of lost revenue. One of my own family members resorted to this issue by working for free for the next few months in order to at least protect her job. Who should be blamed for these kinds of situations is also a point of concern for many people.
“Massive hunger will definitely kill the poor before this virus”, this statement takes my immediate attention. What options are left with the daily-wage earners? I would say no option. They are the ones who are in the most condemning situations. Lockdown for us can be a way to relax or listen to karaoke, but lockdown for them means an end to their life. Unquestionably, this decision by our Prime Minister is the best he could take but when the best turns out to be the worst for someone then I think it’s a moment to question. Going to the internet and writing what’s the best way to spend this Quarantine, should definitely be what’s the best way in which we can help people, but naturally, this would not interest many people. Furthermore, this set off to become more hell when people start with the activity of stockpiling. Unnecessary piling of food is altogether a reason for shortage and ultimately hunger.
My maid Mala who is a housewife and has a small suffering family, I believe could have written this article in a better manner if she would have the relevant education because she is the actual prey to this problem. She being the patient will exactly know the impact this virus can have on our lives because sitting in our homes with a cup coffee can never let us know what an empty cup feels like. Mala said to me think how much we would be tolerating that our Prime Minister apologized to the public for this Lockdown which was certainly a good thing for the masses. What all I could do to help her was to provide her family with some money and eatables. A small contribution with some motivation and a call once in 2 days I believe will not improve her state completely but would certainly provide her with the power to deal with it.  Not only Mala and her family, but there are also millions of people who are in the exact same condition. There are countless Mala in our country who are panicking, enduring and suffering each night with no food in their stomachs. Imagine an exact same situation for yourselves, it would without fail be the hardest situation. This is the impact a virus can create. For sure this article can be of endless pages owing to the fact that there are more stories to 
to this and each story is
painful that the other.
Labor Chowk in Noida has numerous stories of people conveying their problems and asking for help. This Chowk is in the main know for men looking for work. Quite a few men gather there each morning to get some work in order to feed their families with at least a roti but this virus even snatched that from their plates. A very known movie in our Indian cinemas is “Coolie” that says these laborers should get their pay till their sweat dries up. These people anyhow need money each day to feed their families but after these uncontrollable situations they at times are forced to eat filth and muck and this sickness might not kill these people but hunger and stained water definitely will. These stories will never end because it’s all meant to happen but I hope this situation does in order to provide some relief to them.
Also how one would imagine that the movie “Six feet’s apart” would be acted in reality. The solutions related to this pandemic came swiftly and rapidly with the help and support of your doctors and nurses who all were serving us and the sufferers round-the-clock. Solutions like washing your hands for at least 20 seconds correctly, covering your mouth with a face mask and majorly avoiding unnecessary wandering on the streets and keeping yourself away from unhygienic people were asked to follow. The efforts of our doctors, nurses, government officials made me realize that my passion should not be above these notices and announcements but should move hand in hand with it. 
Right at this moment we all are moving with these circumstances and nobody knows what might happen in the future but one thing is of sure that nobody can ever forget this situation and indeed me as it had the ability to knock down my event that day. Many lessons were learned by all the world in these days some being that never ever take anything for granted because if that thing or that person leaves then life tends to become like a rock. Appreciate, admire and cherish whatever you have because you never know when it might leave you, ask from them who have just lost someone from this Corona Virus. Moments to live happily in this world are very few spend it with someone who makes you feel loved. Each day sun rises with a new hope to make a habit to feel the warmth and wish for a healthy life for yourself, your family and this world. Together we all are fighting this virus with unity.
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
