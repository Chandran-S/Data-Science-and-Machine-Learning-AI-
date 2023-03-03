
import requests
from bs4 import BeautifulSoup

# Example URL
url = "https://insights.blackcoffer.com/ai-healthcare-revolution-ml-technology-algorithm-google-analytics-industrialrevolution/"
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
text = ('How AI will impact the future of work?'
'AI experts believe it’s going to be one of the main drivers of the fourth Industrial Revolution and that it has the potential to not just transform the tech sectors and going to open a new chapter of the society of the world that people try to understand themselves better rather than the outside world with AI because people who are naysayer and kind of try to drum up these doomsday scenarios are pretty irresponsible. After all, In the next, five to ten years AI is going to deliver so many improvements and the quality of our lives it is a renaissance, a golden age of machine- learning and artificial intelligence that was the realm of science fiction for the last several decades. AI is probably the most important thing humanities that have ever worked which is more profound than any work with technology, as it is important to harness the benefits and while minimizing the downside is focusing on autonomous systems like self-driving cars seen as the mother of all AI projects and has made applications like self-driving technology viable for the first time, three things happen at the same time number one data collection and data processing became easier because of better technologies right um you need data to fuel AI training and that’s been one of the big drivers the second thing that has happened is that computer processing has become faster that’s like the engine so no matter how much fuel you have if you don’t have that engine and processing the data on a timeframe that’s reasonable was just not possible and the third thing that’s happened is that new algorithms have been developed which has made AI much more powerful so #technology has been changing and developing at a pace that’s much faster than ever before and we have not been used to this rapid pace of change which means that we have not been used to thinking about how it’s going to impact our immediate future. The most important factor responsible for the growth of AI is Google and its AI what Google’s done is given all of us the power to get the relevant information we want at our fingertips this has created a shift in how things are bought but it didn’t happen overnight this started in 2004 but the major change only happens to start 2012 onwards Google’s taken away about 65% of sales people’s jobs that were primarily order takers and the ones that are remaining are likely to be gone over the next decade.'
'In present several AI projects are helping in diagnosing diseases better match up drugs with people depending on what they’re sick they can get treated better so it’s going to help a whole lot of people get treated and get better #healthcare than would have had access to it before if you look at self-driving cars they’re going to be safer than people driving cars and the value that machine learning is providing is actually happening beneath the surface and it is things like improved search results improved product recommendations for customers improve forecasting for inventory management and literally hundreds of other things including speech-recognition or image-recognition that the performance levels are phenomenal  or drug discovery as these biological systems are very complicated because vaccines for TB and HIV developing that’s notably enabled by this rich data advanced in biology and machine learning and recent invention in which is an application we just launched for anybody with visual impairment ass it uses the latest cutting-edge computer vision technology to give anyone the ability to see, so anyone who has dyslexia can now use AI to be able to read better and with the  latest release of Windows 10 has this capability called IJ’s which enable the eye muscle that the gaze can help to type. Like the two sides of the coin, there are negative impacts of AI as well  Bill Gates Ellen Musk also tech giants in a way their views are pessimistic, to say the least, they warned against the potential of AI to replace humans in the workplace and Ella masks even went as far as to claim that AI is the biggest existential threat to mankind. because of the loss of a job, when you think about a job or a career choice if a majority of the tasks that comprise that career choice is likely to be these vulnerable tasks then that is a career at risk in the future so what are the tasks that AI will find? hard to do anything unpredictable anything that requires skills like creative thinking or empathy or interpersonal skills but it’s important to understand tomorrow whether Google is there or not, artificial intelligence is going to progress you know technology has just nature it’s that it’s going to evolve as  technology and in particular AI can, in fact, bring more empowerment more inclusiveness and at the same time it is important to be clear-eyed about displacement and unintended consequences like any other technology and work both skills so that people can find the jobs of the future create new jobs also the policy decisions that help people as they go through this change people already unhappy because of machine learning artificial intelligence as they think  if they’re not innovative enough or not creative enough your job will be taking away by a lot of machines AI for business going to affect the future of work specifically there are jobs that are at more risk of being taken over by AI and automation there is very wide dissonance on this, there are different reports that have been shared by  Oxford study that says 47% of US jobs are at risk of automation over the next few years meanwhile the general population and workers think differently a recent study conducted by college actually identifies that 97% of workers believe that most jobs will be automated but not their own this suggests that the general public needs to be educated on which jobs are  susceptible to this risk which are not and businesses need to be aware of the forthcoming skills gap of course not all jobs are equal the Oxford study that highlights this they examined 700 participants and found the generalist occupations that require creative knowledge or innovation are at least risk the same is true for occupations in education healthcare media and arts jobs on the flip side jobs like telemarketers junior lawyers accountants are at most risk in short there is a simple rule of thumb if your job is in some way predictable or routine the risk of automation is much higher if a job doesn’t require innovation or creativity then the return on investment for companies is higher on machines than real-time employees machines are faster can’t be distracted and can work 24/7 this is actually good for creative marketers because AI and automation can serve to augment their jobs rather than substituting them as impact of emerging technologies on the creative economy they stated that artificial intelligence is changing  creative content from beginning to end by 2030 AI will be able to write high school essays code in Python composed top 40s chart songs and make creative videos but all of these advancements also come with risks and costs take a look at this report by the global Commission on the future of work in the absence of effective transition policies many people will have to accept lower skilled and lower paying jobs high-skilled workers are taking less cognitively demanding jobs displacing less educated workers and this is already happening also technological dividends are being unevenly distributed among firms a very limited amount of companies tend to dominate when it comes to big data just think about Google and Facebook today they alone are responsible for 70% of the referral marketing traffic and  receive more than 50% of total global advertising budget so the question is in businesses workers and social institutions go into the same direction if companies and public policy leaders can understand the evolving landscape they can help the workforce anticipate the upcoming challenges technology and the demographic changes are leading to a smaller workforce compared to the previous generation and a workforce that has to pursue many careers during their time of work we need to provide workers with an environment where they can continuously upskill and grow governments will have to re-evaluate the educational system we will have to continuously learn and grow and companies will have to redesign their structure and their culture around technology just like during the Industrial Revolution we are heading into a new age and the great transformation that we’re about to see by 2022 it is estimated that 20 to 25 percent of the labor force will be displaced within 10 to 20 years however this is also an opportunity for people to get ahead for which different ways have to be find to attract and retain highly skilled workers and allow them the time to up skill themselves even during work hours and it is a  good way  to develop a learning community to benefit from each other and also to use technology to supplement goal tracking and  efforts instead of as a distraction in short what we are doing is  to bridge the dissonance and it is imperative to build a  map of how AI and automation will affect  industry and  company if this is an economic imperative how do people feel about committing itself to a lifelong approach to knowledge as  these risks are important but it is important to do things like from being upfront to have ethical charters like AI safety and to be very transparent and open and how we perceive progress there and figure out global frameworks by which we can engage just like Paris agreement and climate change by using  such forums bring people together as they engage on the hard questions and it will emerge answers and on the question of whether AI is a threat or not, artificial intelligence is not  a threat because there is a rare case where people need to be proactive in regulation instead of reactive because I think by the time we are reactive in AI regulation it’s too late right now we have machine learning algorithms that can solve an incredibly complex problem beyond any human intelligence  as they are mere machines that can be given enormous data set and they come up with brilliant correlations and insights but they’re not going threaten the human population anytime soon because fish intelligent isn’t terrible but human being a smart enough to learn that skills at least to have a complete toolbox to be prepared volatility of the future adaptability.')


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
