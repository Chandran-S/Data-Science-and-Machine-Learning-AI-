
import requests
from bs4 import BeautifulSoup

# Example URL
url = "https://insights.blackcoffer.com/future-of-work-how-ai-has-entered-the-workplace/"
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
text =('Future of Work: How AI Has Entered the Workplace'
'Artificial intelligence and employment are the burning issues nowadays workers need clarity on as we head into the longer term. This article focuses on the various impact of AI on our jobs and explains the benefits of AI in our workplace. That right there must have hit a nerve. However, everything is about to change because this article highlights some of the reasons we should not fear AI.'
'AI is the abbreviation of Artificial Intelligence. Artificial intelligence is often defined as a set of various technologies which will be brought together to permit machines to act with what appears to be human-like levels of intelligence. This includes learning rules required to make certain decisions and reasoning to arrive at certain conclusions, learning from past experiences, and self-correction.'
'AI is of two types.'
'1940'
'Expectations that machines could match humans in terms of general intelligence. By that, we mean machines could have the capability to find out, Reason, and React.'
'1950'
'Alan Turing develops the Turing Test; a test to determine whether a machine is intelligent.However, it wasn’t for another 60 years or so that any program was deemed to have passed.'
'1955'
'The first time a computer virus defeats a person’s World Champion during a parlor game.'
'1956'
'John McCarthy invents the new term ‘Artificial Intelligence’ when he held the primary Academic Conference on the subject of AI.'
'1962'
'Arthur Samuel‘s machine learning checkers (draughts) program, beat a checkers master.'
'1980'
'Reinforcement Learning is introduced. This is a kind of programming that uses rewards and punishments to coach a machine to interact with its environment.'
'2012'
'A research group led by Geoffrey Hinton wins the Image Net competition – this competition requires AI to categorize about 1.2 million images into any of 10,000 different categories. The level of accuracy was adequate to that of the typical human completing an equivalent task manually.'
'2014'
'Eugene Goostman’s chatbot, a bot pretending to be a 13-year-old boy, supposedly passes the Turing Test, a test which nobody has passed before! But controversy arose with this claim as:'
'1. Experts claim it only lasted five minutes.'
'2. It had been deemed biased as Eugene’s mother tongue (Ukrainian) wasn’t equivalent to the judge’s (English), which is a plus as language is one among the few ways we will tell the difference between a person’s and machine.'
'2018'
'Alibaba’s AI Model performs better than humans during a reading and comprehension test at Stanford University, scoring 82.44 against the 82.304 scored by humans!'
'Artificial Intelligence (AI) is here to remain, and lots of people aren’t happy. After all, it’s hard to embrace something that would displace about 40 percent of human jobs within the next 15 years. In an interview for CBS’s hour, Kai-Fu Lee (a Chinese AI expert) also mentioned truck drivers, chauffeurs, waiters, and chefs as a number of the professions that will be disrupted.'
'But if you were to ask the experts, they might unwaveringly confirm that no matter all the noise, AI is here to profit us all. Case in point, an executive briefing by the McKinsey Global Institute revealed that AI and automation are creating opportunities for the economy, society, and business.'
'That said, it’s time to repress the widespread idea of artificial intelligence taking jobs. So, let’s highlight a number of the useful developments you’ll expect from this technological phenomenon.'
'While the relationship between artificial intelligence and jobs is a matter of hot debate, it is still safe to say that AI will indeed offer new opportunities. According to the planet Economic Forum report, robots and AI will create as many AI jobs as they displace. This conclusion is entirely viable as it is easy to identify some of the many careers in artificial intelligence, for example, data scientists, who evaluate the decisions made by AI algorithms to eliminate any biases. Apart from that, some other AI occupations include:'
'Transparency analysts: people tasked with classifying the varied sorts of opacity for algorithms. Smart-system interaction modelers: experts who develop machine behavior based on employee behavior. Machine-relations managers: people that champion the greater use of algorithms that perform well. As far because the competition for jobs between humans and robots goes, worth noting is that there are jobs that AI can’t replace. Roles that need leadership, empathy, and delegation are samples of the various jobs that are safe from automation.'
'Automation will stir positive change in the workplace. When AI is employed during recruitment or maybe performance management, all workers are going to be evaluated in an unbiased, fact-based manner. In turn, Human Resources managers can get to consider other essential strategic undertakings that ensure balance within the workplace.'
'AI can help HR departments to use machine learning (ML) in discovering where issues like bias stem from and assist them to act accordingly faster. ML shines in identifying instances of bias. In turn, this may promote fairness and variety within the work setting.'
'The impact of AI in business is already felt, and this is often expected to continue through to the longer term. A few years from now, AI-oriented architecture is forecasted to require the lead in assisting businesses to hold out operations in additional comprehensive ways, thus shifting them from traditional data science and machine learning models. It will be necessary to maneuver to business outcomes because AI will play an important role in multiple aspects of the business. While there is no way of telling the future of artificial intelligence in business for sure, it makes sense for owners to keep up with the evolution to avoid being left behind.'
'The workforce of the longer term will lean more towards innovation and creativity. Businesses have spent higher a part of the previous couple of years studying AI automation and the way they will leverage it to realize results fast. With statistics showing that workers spend up to 40 percent of their hours at work performing repetitive tasks, every business should consider automating any functions which will be automated. Automation is not new; machines have been replacing human labor in different areas for decades.'
'However, within the coming years, mundane daily tasks will become fully automated. Already, 39 percent of organizations were completely reliant on automation in 2018. With repetitive tasks taken care of, employees can focus their energies on high-value customer-oriented tasks and collaboration. The designs of workplaces and workflows will also change with the implementation of AI technology. More people will begin to figure more closely with machines as companies will strive to become more agile.'
'Companies that implement AI in their business strategy will experience dramatic improvements in their customer experiences, and their employees are going to be more motivated. Encouraging creativity rather than the performance of repetitive tasks gives workers more fulfillment in their jobs as well.'
'With the web of things and AI working hand in hand, identifying trends and solving problems within the business world will become more convenient and also sustainable. AI, alongside other technologies, is predicted to vary the planet by impacting the way businesses run. With time, we’ll be ready to combine both human and artificial brains to seek out solutions to major global problems.'
'It will even be easier to foresee problems with more accuracy and nip them at the bud with the help of AI. But these positive impacts can only be felt if stakeholders are transparent and mindful in their use of the technology for the greater good of everyone else.'
'According to a report by PWC, 54 percent of companies confirm that the implementation of AI-driven solutions in their companies has already improved productivity. AI and automation, even once they are implemented partially, have unlimited potential for any business. Workers’ skills, attitude, training, command chain, and workflows protocols are a number of the leading productivity challenges that companies face.'
'Apart from increasing productivity, AI systems will help businesses to chop down on costs, improve the standard of their products or services, and make better customer profiles. As a result, companies also will get higher profits, which may be shared among stakeholders as dividends, or reinvest it back within the business. Improved productivity also means firms are going to be ready to sell their products at lower prices, thereby creating more demand among customers and more job opportunities for workers. Businesses can, therefore, use human labor to take up those jobs that have been created by AI and cannot be automated.'
'Artificial Intelligence isn’t showing any signs of slowing down. Soon, it’ll become another necessity of life, a bit like the web. But for now, more and more businesses are beginning to realize how invaluable AI automation and data interpretation are. Though machines will take some jobs initially, the roles created by automation will also keep soaring within the next few years.'
'Will, we’ve reached the purpose of accelerating human intelligence by artificial means? Who knows? What we all know, however, is that those that are going to be sought-after in AI and employment are individuals who have the relevant skills. There’s work that not even the machines can do; so, we should make ourselves as valuable as we can be in our field and we will be irreplaceable.')

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

