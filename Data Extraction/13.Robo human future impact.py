
import requests
from bs4 import BeautifulSoup

# Example URL
url = "https://insights.blackcoffer.com/ai-human-robotics-machine-future-planet-blackcoffer-thinking-jobs-workplace/"
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
text = ('How Robo Human will Impact the Future?'
'It’s the year 2060. An automaton in a Research Laboratory says to a Scientist, “Warning! Error Occurred Reformatting Hard Disk Now!” The scientist panics. Automaton says again,” Ha! Ha! Just Kidding! “.'
'Funny Right. Before some of you say that this joke isn’t realistic, “How can an Automaton tell you a joke?” But what if I tell you in 2017, “Sofia” the robot made a joke on the show Good Morning Britain! Who thought computers could tell us a joke? Hard to believe? Well, the idea of giving computers human-enjoy thinking has now become a reality. Thanks to the technological advancement in AI in the last decade.'
'Before diving deep into how AI can impact the future of work, let’s begin with the simple question: what’s AI? Artificial Intelligence provides machines the power to think from data. The machine uses the patterns and trends found in data and makes its decision, but cannot create thought beyond these patterns and trends.'
'With the rise of AI, humans are divided into one question. Are machines human’s friend or foe? Tech executives and politicians on conference stages, campaign rallies, and even science fiction Hollywood movies like Carbon Black, Westworld, Minority Report, and Ex Machina have given their take on this question. Some believe AI will help us solve problems while others believe that the rise of AI will result in destruction and maybe the end of the world, we all know.'
'Stephen Hawking made it no secret of his concern about the rise of superhuman AI that eventually would escape earth to a new planet. No, this isn’t a plot of Black Mirror. Right now, Superhumans may not be a reality, but AI is.'
'“Homo Deus”, the emergence of the new Digital God using AI. God must also worry, as AI might take his job.'
'Here’s some Career Advice, have you thought about being a Robot? The fear that AI would automate all jobs in the future eventually leaving all humans jobless has been daunting for many workers today. Statistics show that nearly 37% of workers worry about losing their jobs to robots. While another thought that many people believe is that though the rise of AI with result in automating most of the jobs in the future, however, it also will create millions of new job opportunities.'
'AI is already replacing most manual and repetitive tasks. For example, buying a metro ticket or a movie ticket is now almost a human-less interaction. Each year the number of industrial robot jobs increases by 14 %. At this rate, it’s predicted that the 20 million jobs in the manufacturing industry will be replaced by robots due to automation.'
'The coronavirus pandemic and recession have boosted the demand for automation. The Robotic Process Automation (RPA) Software industry has experienced an increase of 19.53% in the year 2021. Coronavirus pandemic has increased interest in technology that reduces human contact as minimal for making workplaces safe.'
'Our workplaces will look much different in the next five to ten years. AI will help humans in simplifying repetitive processes. The two most important catalysts for the future of work are the two D’s- Digitization and Datafication. Digitalization is converting data to digital formats (computer-readable). For example, text to Html, analog video to YouTube video. Digitization helps in increasing data exponentially. Datafication is quantifying human life to data and improving the data-driven business model. By 2025, it is forecasted that the digital transformation space will build in a $3,294 billion industry!'
'One thing is clear, no data, no future of work. What we find is that the future of data and the future of work will go hand in hand. The total volume of data in the datasphere that is created, captured, copied, and consumed in the world is predicted to reach 175 zettabytes by 2025. To give you a much better picture for understanding, if we represent the digital universe as stacks of tablets, there would be 27.25 stacks from earth to the moon.'
'It’s time to prepare for the data-dominated future as Industry 4.0/Fourth Industrial Revolution has begun. So, let’s see how artificial intelligence will affect the following fields:'
'Human Resource: Nowadays, recruiters use AI-powered tools for hiring workers. Using these tools, recruiters get insights into a candidate’s skills, personality and even check whether the candidate is fit for the organization. For example, the company AllyO first identifies high-potential candidates through assessment and smart screening, and then automatically schedules interviews using AI. HR departments at large companies receive hundreds of resumes for a job opening. Entry-level roles focusing on screening and scheduling will be automated. AI will automate specific HR jobs, not HR roles. A Deloitte study found that AI has already eliminated 800,000 low-skilled jobs in the UK, but 3.5 million new jobs were also created. Roles that focus on complex decisions like resolving disputes within a department will continue to be a very human endeavor.'
'Finance and Accounting: In 2015, a report from Accenture named “Finance 2020: death by digital” predicted that 40 percent of transactional accounting work would be automated by 2020. Has technology replaced the human factor? Well, AI has created new jobs involving managing the AI system and using the information to create insights. For example, accounting software has already automated bookkeeping tasks that used to be done by humans, but that’s only opened the door for former bookkeepers to learn skills needed to run and manage the software for employers and clients. Advisors are another crucial role of the accounting and finance team. Using the information gained from transactions in books, the team creates insights to improve business strategy. Owing to automation, the team spends more time analyzing numbers.'
'Marketing and Sales: Marketing automation has helped companies strategize the proper utilization of the company’s resources, managing time, and achieving budget targets. Marketing automation has helped to draw conclusions at a scale no marketer ever would. In this process, marketers and machines both excel in different parts. Marketers using AI tools drive more conversions in less time. Human Intelligence with technology can help identify the right customers to talk to and at the right time. Modern Marketers understand the insights from any marketing campaign and create it into effective messaging.'
'Engineering: Technology is changing in a blink of an eye. The technologies used five years ago in the industry have become obsolete today. Engineers will have to keep up with the technological advancements and keep upgrading their skills to stay relevant in the industry. Learning to work alongside machines and designing work such that interaction better humans and machines are better are going to be important skills for engineers in the future.'

'In the 18th and 19th centuries, the rise of the industrial revolution centuries led to millions of people losing their jobs because of scientific advancements. But that also ended in creating millions of other jobs. Statisticians have said, when automation destroys jobs, people find new ones. Thus, AI holds a more optimistic picture for the future.'
'In the future, AI is not going to replace humans, rather make jobs more humane. AI will disrupt millions of middle and entry-level jobs in the next few years but will also create millions of additional jobs and help to boost economies.')
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

