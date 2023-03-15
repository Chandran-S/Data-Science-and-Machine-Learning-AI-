
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/how-you-lead-a-project-or-a-team-without-any-technical-expertise/'


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
text = ('''How you lead a project or a team without any technical expertise?
Have you ever wondered what’s common between Sir Alex Ferguson, Pep Guardiola, and Jose Mourinho? I know it might look a lame question, especially to the ardent football followers. At first instinct, the answer is that they are regarded as the top managers in the football universe. #UCLwinners. But when given a closer look, these managers, weren’t that great when it came to their careers as football players. Any naïve observer can build a fallacy of causation and correlation that those who don’t have a great career as a football player will go on to become a great football manager. But is this restricted just to football? Mike Brearley, one of the greatest captains in the history of cricket and the author of the book the art of captaincy, wasn’t the prolific batsman of his time. Our former President, A P J Abdul Kalam didn’t have a political background, but was still appointed as the president of our country and went to become one of the finest presidents our country has ever had. So, can it be stated with confidence that great leaders need not be the expert in their respective fields?Let’s understand the leadership role, types of leaders, and explore a few more perspectives before we arrive at a conclusion. 
Let’s start by first understanding what or who is a leader. A leader is not a position or a designation. It’s a virtue if employed correctly. From a janitor to the CEO, from a student to a researcher, from a content writer to a philosopher everybody is a leader. (#leadersareeverywhere) So, what are the virtues of a leader? One of the famous personality trait theories, the Big 5 personality trait theory is always linked closely with leadership. As per the various researches, the personality traits of extraversion, openness, and conscientiousness are associated with leaders more often than the other two traits. Not just these humble and empathetic are also considered as the adjectives that are frequently associated with leaders. But nowhere is technical expertise related to a leader. #NOTSOTECHNICAL
As stated by Colin Powell Leadership is the art of accomplishing more than the science of management says is possible. #It’sART If leadership is an art and leaders are the artists then more or less the leaders can be viewed as directors of the films who are organizing, arranging, managing, and directing the scenes. Another analogy for that can be the orchestra where the conductor is using the baton/stick to set the rhythm of the band. So, is it necessary for the conductor to himself or herself by a great violinist? No. When it comes to the corporate sector the leader need not be an expert on the technology but should know how the technology functions so that he/she can utilize it in the best possible way. They can allocate the task of building technology to the right technical experts. And how do the leaders identify the right person for that job? Or how do leaders know what qualities to look for in a person? #Experience. Every great leader once started off as one of the frontline workers or at the bottom of the pyramid in the hierarchical structure. Even though they might not have had the technical expertise in their respective fields, their visionary outlook, their management skills, and the experience they gathered helped them in achieving their targets. Leaders can always take the assistance of the subject matter experts to guide them in situations of technical difficulties. As per the great man, situational and trait theory of leadership, leadership is innate to the person. Leadership though can be conditioned or can be brought out in a person by rewarding him, great leaders show unconditional attachment towards their vision. Everyone knows how to read a map, but which path to choose isn’t something that can be taught. Leaders give the direction or vision to the company and drive it closer to the target that the company wants to achieve. Leadership theories don’t focus on the technical knowledge of the leaders, but rather more on people skills, their approach towards a problem, and how well they guide and hire others to get the job done, which is also evident as per the managerial skills required as per Robert Katz.
Though it can be argued that technical experts are also the great leaders in their areas, but these leaders are generally at the middle management level. As per the different leadership styles, a servant leader is the one who always tries to achieve the goal of the team. These leaders are the ones who have a management objective laid out in front of them and they drive the team closer to achieving the target by having the people-first mindset, a collaborative approach towards solving the problem. These are the people or the leaders needed at the execution level of the project or those who are too close to the employees at the bottom-most level. Since the managers at the top of the pyramid don’t get much time to invest in people they appoint the right people on their behalf to get the job done.  
Consider an IT firm. So as per the points mentioned above, does it mean that a leader or the manager shouldn’t know how to code? Depends. If the person is the first line manager then he or she should probably know how to code so that he or she can help the team in achieving the target of producing less garbage which might be one of the project’s KPIs. The project manager, who might be an operations manager, (#MBA) from an IT background, probably need not know what line of code is to be written but should know the algorithms or the logic behind the code so that he or she can validate the code from the client’s perspective. The CEO, might not be from an IT background but should know how to run an IT firm rather than the syntax of the code.
But in today’s competitive world leaders can’t be one dimensional. They need to have expertise in multiple areas to be at the top. Mike Brearley if got a chance to play a T20 match, won’t even get picked into aside, not because of his age, but because of his not so great looking batting statistics(#Thegamehaschanged). His leadership skills won’t be enough to get him into the squad. But he can be a great leader if selected to coach the top 15 captains in the world today. As we have already discussed the film directors, their job isn’t to teach the cameramen on how to set up or operate (lens settings, etc) the camera. Their job is to guide the cameramen on what angle it should be held and how he or she should coordinate with the lights crew so that a perfect shot is captured. So even if a leader may not be a technical expert of his field, but if he or she has the right approach and mindset, understands the system and its process, then he or she can hire the best talent for the job and can get the work done by guiding the talent, with his or her vision.
So, the answer to the question, which we came across in the first paragraph, is yes and no. Depending upon the position in the hierarchy (yes for a front-line manager and no for a CXO) and the type of the leader you are, the level of technical expertise needed varies. Though as stated earlier by Robert Katz, the CXO’s need to have the least amount of technical expertise, the CXO should know how the technology functions or what are the applications. What processes are to be followed to build the technology.  He or she should now what is to be done rather than how it is to be done, (#itsnothowitswhat). Otherwise, even though the company might be sitting on a gold mine but If the leader isn’t aware of that or isn’t learned enough to know what to do with it, no company can succeed even with having all the resources at their disposal. Leaders command respect and don’t demand it and that’s possible when they give value to their front-line employees, seek timely advice from the technical experts, and put the resources to the best of its uses. Otherwise, you will be surrounded by technical experts like dilberts. (#dilbert)
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

