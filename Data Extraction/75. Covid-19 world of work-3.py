
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/how-will-covid-19-affect-the-world-of-work-3/'


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
text = ('''How will COVID-19 affect the world of work?
A respiratory illness that started from patient zero in Wuhan, a city in China, has now engulfed the entire globe in its clutches. With over 2 million confirmed cases and deaths north of 100,000, Coronavirus, better known as COVID-19, is a challenge unfathomed. History is earmarked with infections and diseases that ravaged continents time and again—Black Death decimated Europe by claiming 200 million lives between 1347 and 1352, Smallpox killed 500 million over decades, bouts of Cholera outbreaks, the emergence of HIV/AIDS and what not—the list is huge and intimidating. But COVID-19 is cut from a different cloth, exposing various loopholes and the modern world’s inability to fend off pandemics and epidemics despite advancements in Science, healthcare, and technology.
The COVID-19 pandemic has invoked an unprecedented seismic shift across segments—the world economy is declining, third world countries with underdeveloped healthcare industry are spiraling into poverty and darkness (read about Ecuador being trapped between death and debt), the IMF declaring a global recession—it’s a concoction of tragedy, turmoil, and tribulations. One particularly interesting domain to mull over is the world of work, an umbrella term that encompasses industries, businesses, organizations, and all that jazz.
Hundreds of thousands of layoffs are happening in sectors like travel and tourism, hospitality, manufacturing, and basically every other customer-facing domain; not to mention those which rely solely or heavily on physical labor. Quarantine, a measure proposed and already implemented by hundreds of nations to limit the spread of coronavirus, has stopped people from going to their workplace. Offices are empty, factories are silent with machines gathering dust, hotels and restaurants are deserted, etc. The outcome has compelled organizations to remain strong in the face of this adversity and circumvent the problem of employees not being able to step out of their homes. The concept of work from home or WFH is now mainstream and the irrational ideology of decline in productivity and revenue upon implementing WFH has gone for a toss. The surge in the use of video conferencing apps attests to this. Zoom, a popular video call software, has witnessed more than double the number of users in March. Organizations are now cognizant of WFH and its benefits.
The onslaught of ‘normal’ working conditions by COVID-19 has made companies revisit the concept of redundancy. It is time to re-evaluate one’s workforce and underline core jobs essential to keeping the company running while highlighting the not-so-important ones. Of course, to render employees jobless and terminate them at this point casts a negative light on the organization’s reputation. So, companies are smart enough to ward such drastic cost-cutting measures at the moment—a few that did so are under the radar of their respective governments for the wrong reasons. It’s not unreasonable to expect corporations to impose significant changes in workforce strength and hierarchy once economic rectitude kicks in.
With passing time, more and more countries now have lockdown measures in place. Confined within homes, those with jobs now see the term ‘work-life balance’ unravel. The vicious cycle of work—wake up, go to the office, back at home, have food, sleep, and repeat the same all over again—is a hot-button topic today. The disastrous consequences of burnout at work, job-related stress and the consistent pressure to perform are at the receiving end of much-needed attention like never. Often neglected aspects of life like health, family, contentment, and well-being are not taken for granted anymore. The canard that coronavirus and its virulence is restricted only to the aged has been put to rest. With fatalities in every age-group, workers now grasp the importance of neglecting health for office. Once life’s back on track, a reduction in work-related burnouts and health issues is imminent.
Boredom and restlessness (brought about by lockdown) have compelled people to upskill themselves; the presence of MOOC and online courses has buttressed e-learning/online education. LinkedIn is now riddled with professionals flaunting the latest certification from Udemy and Coursera. It took a pandemic for many employees to correlate continuous learning with success and recognition at the workplace. After a couple of months, you’ll find yourself surrounded by smart peers whom you didn’t heed to earlier.
Under the pretext of pursuing long left hobbies and passions, the number of active freelancers has increased. Individuals pressed by office workers are using the extra time at hand to take up freelancing and augment their primary source of income. Content writers, copywriters, graphic designers, tutors are now dime a dozen. Will freelancers sustain this flow once the health scare settles? We don’t know. But, COVID-19, without any second thoughts, has given a leg-up to the freelancer culture.
Working across time zones has always been a bummer. Converting Indian Standard Time to Eastern Time Zone, Central Time to some other time zone, etc. for meetings meant ensuing headaches. To ensure the business continues, employees divided by geographies are now more accommodating to get cross-functional tasks done. Organizations are witnessing better coordination and collaboration across teams sitting in different parts of the world.
Car manufacturers are developing ventilators, hundreds of tech ventures are building COVID trackers, predictors, assistant chatbots and awe-inspiring apps, logistics are bending accepted ways to deal with transportation—an influx of change for good lies in front of us.  Unimaginable innovation and creativity have risen from the depths of oblivion!
The dark side of social media i.e. the use of such platforms to spread rumors and false news is has become more evident than ever. An Indian IT giant expelled one of its employees in lieu of his abominable Facebook post to spread Coronavirus. Organizations will heighten vigilance of what their staff post, publish, or advocate on digital platforms. Condemnable information would bring in dire consequences that extend beyond simple termination. It seems social media has taken a hit for good.
All in all, the global outbreak of COVID-19 has ushered in a slew of radical changes to how work is perceived in the grand scheme of things. 2020 will go down in history books as the year that shook the very fundamentals of the world of work!
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

