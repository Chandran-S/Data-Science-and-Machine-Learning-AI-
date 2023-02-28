
import requests
from bs4 import BeautifulSoup

# Example URL
url = "https://insights.blackcoffer.com/will-ai-replace-us-or-work-with-us/"

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
text = ('''Will AI Replace Us or Work With Us?
“Machine intelligence is the last invention that humanity will ever need to make”
Nick Bostrom
To put it frankly, Artificial Intelligence will eventually replace jobs. Workers in a variety of industries, from healthcare to agriculture and manufacturing, should expect to witness hiring disruptions as a result of Artificial Intelligence.
If history has taught us anything, it is that disruptive paradigm-shifting business ideas not only make a fortune for the innovators, but they also build the groundwork for new business models, market entrants, and job opportunities which will inevitably follow. It is true that robots today or in future will eventually replace humans for many jobs, but so did innovative farming equipment for humans and horses during the industrial revolution. But that does not mean that our jobs as humans will end here. We, on the other hand, will be required to generate and provide value in whole new ways for entirely new business models as a result of these changes.
According to 71% of the businesses worldwide, Artificial Intelligence can help people overcome critical and challenging problems and live better lives. Artificial Intelligence consultants at work will be more or equally fair, according to a whopping 83% of corporate leaders. These results demonstrate that Artificial Intelligence is steadily extending its measures, yielding societal benefits and allowing citizens to live more fulfilling lives.
Since the advent of Industry 4.0, businesses are moving at a fast pace towards automation, be it any type of industry. In 2013, researchers at oxford university did a study on the future of work. They concluded that almost one in every two jobs have a high risk of being automated by machines. Machine learning is responsible for this disruption. It is the most powerful branch of artificial intelligence. It allows machines to learn from data and mimic some of the things that humans can do.
A research was conducted by the employees of Kaggle wherein an algorithm was to be created to take images of a human eye and diagnose an eye disease known as diabetic retinopathy. Here, the winning algorithm could match the diagnosis given by human ophthalmologists. Another study was conducted wherein an algorithm should be created to grade high school essays. Here too, the winning algorithm could match the grade given by human teachers.
Thus, we can safely conclude that given the right data, machines can easily outperform human beings in tasks like these. A teacher might read 10,000 essays over a 40-year career; an ophthalmologist might see 50,000 eyes but a machine can read a million essays and see a million eyes within minutes.
Thus, it is convenient to conclude that we have no chance of competing with machines on frequent, high volume tasks. 
Tasks where machines don’t work
But there are tasks where human beings have an upper hand, and that is, in novel tasks. Machines can’t handle things they haven’t seen many times before. The fundamental rule of machine learning is that it learns from large volumes of past data. But humans don’t; we have the ability of seemingly connecting disparate threads to solve problems we haven’t seen before. 
Percy Spencer was a physicist working on radar during world war 2 where he noticed that the magnetron was melting his chocolate bar. Here, he was able to connect his understanding of electromagnetic radiation with his knowledge of cooking in order to invent the microwave oven. Now this sort of cross pollination happens to each one of us several times in a day. Thus, machines cannot compete with us when it comes to tackling novel situations. 
Now as we all know that around 92% of talented professionals believe that soft skills such as human interactions and fostering relationships matter much more than hard skills in being successful in managing a workplace. Perhaps, these are the kind of tasks that machines can never compete with humans at. 
Also, creative tasks: the copy behind a marketing campaign needs to grab customers’ attention and will have to stand out of the crowd. Business strategy means finding gaps in the market and accordingly working on them. Since machines cannot outperform humans in novel tasks, it will be humans who would be creating these campaigns and strategies. 
Human contact would be essential in care-giving and educational-related work responsibilities, and technology would take a backseat. Health screenings and customer service face-to-face communication would advocate for human contact, with Artificial Intelligence playing a supporting role. 
So, what does this mean for the future of work? The future state of any single job lies in the answer to one single question: to what extent is the job reducible to tackling frequent high-volume tasks and to what extent does it involve tackling novel situations?
Today machines diagnose diseases and grade exam papers, over the coming years they’re going to conduct audits, they’re going to read boilerplate from legal contracts. But does that mean we’re not going to be needing accountants and lawyers? Wrong. We’re still going to need them for complex tax structuring, for path breaking litigation. It will only get tougher to get these jobs as machine learning will shrink their ranks. 
Amazon has recruited more than 100,000 robots in its warehouses to help move goods and products around more effectively, and its warehouse workforce has expanded by more than 80,000 people. Humans pick and pack goods (Amazon has over 480,000,000 products on its “shelves”), while robots move orders throughout the enormous warehouses, therefore reducing “the amount of walking required of workers, making Amazon pickers more efficient and less exhausted.” Furthermore, because Amazon no longer requires aisle space for humans, the robots enable Amazon to pack shelves together like cars in rush-hour traffic.” More inventory under one roof offers better selection for customers, and a higher density of shelf space equals more inventory under one roof.
Kodak, once an undisputed giant of the photography industry, had a 90% share in the USA market in 1976, and by 1984, they were employing 1,45,000 people. But in the year 2012, they had a net worth of negative $1 billion and they had to declare bankruptcy. Why? Because they failed to predict the importance of exponential trends when it comes to technology. On the other hand, Instagram, a digital photography company started in 2012 with 13 employees and later they were sold to Facebook for $1 billion. This is so ironic because Kodak pioneered digital photography and actually invented the first digital camera but unfortunately thought of it as a mere product and didn’t pay attention towards it and this created the problem.  
We live in an era of artificial intelligence (AI), which has given us tremendous computing power, storage space, and information access. We were given the spinning wheel in the first, electricity in the second, and computers in the third industrial revolution by the exponential growth of technology.
Airbnb, which is a giant start-up and is known for enabling homeowners to rent out their homes and couches to travellers, for example, “is now creating a new Artificial Intelligence system that will empower its designers and product engineers to literally take ideas from the drawing board and convert them into actual products almost instantly.” This might be a significant breakthrough whether you’re a designer, engineer, or other type of technologist.
Differences that Automation brings onto the table: 
There are three key changes that automation can bring about at the macro level: 
Artificial Intelligence isn’t just a fad. Tractica, a market research firm, published a report in 2016 that predicted “annual global revenue for artificial intelligence products and services will expand from 643.7 million in 2016 to $36.8 billion by 2025, a 57-fold increase over that time span.” As a result, it is the IT industry’s fastest-growing segment of any size.”
The reduction in need for people as a result of Artificial Intelligence and related technologies, which resulted in job layoffs, was a cause of fear. In India alone, job losses in the IT sector have reportedly reached 1,000 in the last year, owing to the integration of new and advanced technologies like artificial intelligence and machine learning. 
Most of the IT companies such as Infosys, Wipro, TCS, and Cognizant have reduced their employee base in India and are recruiting less, while engaging more personnel in the United States and investing heavily in “centres of innovation.” Artificial Intelligence and data science, which are currently the trending aspects that require fewer people and are primarily located abroad, aren’t helping the prospects of local employees. Another factor is that the computer industry is continuously growing and would develop to a size of two million workers. Unfortunately, it’s a drop in the bucket compared to what robots are doing to Information Technology’s less-skilled brothers. 
Large e-commerce sites that used to be operated by armies of people are now manned by 200 robots produced by GreyOrange, which is an Indian company based out in Gurgaon. These indefatigable robots lift and stack boxes 24 hours a day, with only a 30-minute break for recharging, and have cut employees by up to 80%. For efficiency, this is a victory but a disaster for job prospects. 
Internal re-skilling and redeployment of staff is a critical requirement of the hour. Artificial intelligence has presented Indian policymakers with epistemological, scientific, and ethical issues. This requires us to abandon regular, linear, and non-disruptive mental patterns. The tale of artificial intelligence’s influence on individuals and their occupations will only be told over time. It is up to us to upskill ourselves and look for ways to stay current with the industry’s current trends and demands. 
So, will machines be able to take over many of our jobs? The answer is a resounding yes. However, for every job that is taken over by robots, there will be an equal number of positions available for people to do. Some of these human vocations will be artistic in nature. Others will necessitate humans honing superhuman cognitive abilities. Humans and machines can form symbiotic relationships, assisting each other in doing what they do best. In the future, people and machines may be able to collaborate and work together towards a common goal for any business they work for. 
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

