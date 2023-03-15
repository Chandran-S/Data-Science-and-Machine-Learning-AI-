
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/what-are-the-key-policies-that-will-mitigate-the-impacts-of-covid-19-on-the-world-of-work/'



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
text = ('''What are the key policies that will mitigate the impacts of COVID-19 on the world of work?
From Alibaba to Ping An and Google to Ford, companies around the globe are telling staff to figure from directly from home in a bid to stem the spread of COVID-19. Such remote functioning at scale is unprecedented and can leave a long-lasting impression on the way people live and work for several years to return. China, which felt the primary impact of the pandemic, was an early mover during this space. As home to a number of the world’s largest firms, it offers lessons for people who are just commencing to embrace the shift. Working from home skyrocketed in China within the wake of the COVID-19 crisis as companies told their employees to remain home. Around 200 million people were working remotely by the tip of the Chinese New Year holiday. While this arrangement has some benefits, like avoiding long commutes, many employees and firms found it challenging. One employee at a web company quipped his workday changed from ‘996’ to ‘007,’ meaning from nine to nine, 6 days every week, to all or any the time. On the private front, employees found it difficult to manage kids’ home-schooling via video conference while coordinating with remote colleagues. At an organization level, many felt that productivity rapidly tailed off if not managed properly. If done right, remote working can boost productivity and morale; done badly, it can breed inefficiency, damage work relationships, and demotivate employees. These kind of things are happening due to the impacts of COVID-19 on the world of work
Teams or whole business units working remotely can quickly end in confusion and a scarcity of clarity. Being isolated ends up in uncertainty about who to speak to on specific issues and the way and when to approach them, resulting in hold-ups and delays. That’s why establishing a structure and architecture for deciding and effective communication is vital. Here, smaller cross-functional teams are helpful, each with a transparent mission and reporting line, where directions and tasks are easy to implement. This also simplifies onboarding new hires, who can integrate faster in a very tight-knit group, at a time when the broad sweep of the organization isn’t visible or easy to feel. With fewer in each team, there’s longer to urge to understand one another and build the trust that will grow more organically within the office. At Ping An Insurance, workers are typically grouped in project teams of, at most, 30 members, while larger business units are divided to assist them to stay agile. Strong company-wide foundations underpin this, like having a typical purpose and unified goals. Providing clarity on what decisions to escalate and which of them is tackled at the team level helps drive progress. To mitigate the consequences of closed retail stores, one leading fashion company founded a technique room and redeployed staff into four cross-functional squads to support its front-line. It designed standard ways for live broadcasting and established internal best practices to encourage front-line staff to use new retail tools to drive sales remotely.
 The lesson: putting in small, cross-functional teams with clear objectives and a typical purpose keeps everyone on the identical strategic course.
Managing people is one of the foremost difficult elements of remote working, not least because everyone will respond differently to the cultural shift and challenges of the home-working environment. Leaders have to energize the entire company by setting a transparent direction and communicating it effectively. Offering a powerful vision and a sensible outlook can have a strong effect on motivation across the organization. It’s essential to foster an outcome-driven culture that empowers and holds teams answerable for getting things done while encouraging open, honest, and productive communication. Empowering your team in this way pays dividends.WeSure, a part of leading internet company Tencent, assembled a COVID-19 response team5 at the beginning of the year to supply amount, freed from charge to front-line medical workers.
Alan Lau, CEO of WeSure, credited his team, saying that they had worked nonstop, many from remote locations while inactive during the Chinese New Year break, demonstrating how responsive they were to the vision.
For managers, the challenge is to guide, inspire, and direct their team in their daily course of labor, while being physically remote. Upping the amount of interaction can even work well here. One chief information officer, responding to a McKinsey survey, said he’s texting the whole company with regular updates because it’s a more human way of communicating than via the official corporate channels. When working within distributed teams, e-commerce giant Alibaba increases the frequency of its one-to-one communications with employees to a weekly basis and, in some teams, members submit a weekly report for his or her colleagues, complete with plans for the week ahead. Alibaba’s productivity app DingTalk (Ding Ding) has features inbuilt to facilitate this by allowing managers to send voice-to-text messages to their teams, and to test in on progress.
The lesson: Determining how you communicate is simply as important as what’s being said, and it has to be done confidently, consistently, and reliably
As companies transition to the new normal, it’s important to acknowledge that some employees are also facing other pressures reception, including caring for his or her children when schools are shut, resulting in feelings of isolation and insecurity. Business leaders must respect and address these additional needs. Empathy may be a crucial tool here, offering how to attach, promote inclusiveness, and make a way of community in a very barren of physical interaction. Increasing social interactions within the team, particularly through one-on-one catch-up, guards against feelings of isolation and demoralization and creates space for people to talk up and share their thoughts. By creating a way of psychological safety for his or her colleagues, being inclusive in deciding, and offering perspective in challenging moments, managers can stay closer to what’s occurring, surface issues, and help their teams solve problems effectively. A similar approach is very important when coping with customers and clients, providing valuable stability and enabling them to navigate unknown waters confidently. As an example, one global bank asked their relationship managers to attach with small-business customers via WeChat and video calls to know their situation and help them weather the crisis. To try and do so effectively at scale, the managers are supported through dedicated product programs, online articles, scripts for communicating with clients, and internal training. Inclusion is that the ultimate show of empathy. Creating outlets for sharing best practices, success stories, challenges, and water-cooler chat are significant to making a person’s connection. Giving employees space to pursue personal or social endeavors, providing a transparent span of control, and assigning meaningful tasks also can spur motivation. 
The lesson: Connecting on a private level and instilling empathy within the culture is doubly important when working remotely.
Moving to remote working risks disrupting the office-based flows and rhythms and it will be easy to hit the incorrect note or miss important virtual meetings thanks to packed schedules. Spend time along with your team addressing the nuts and bolts of how you’ll work together. Cover the daily rhythm, individual constraints, and specific norms you’ll decide to and anticipate what might get it wrong and the way you’ll mitigate it. How companies plan and review their workflows has to change to reflect this. The challenges of the new working pattern and of not being in one room together will be overcome by creating a digitally facilitated cadence of meetings.
One leading insurance firm adopted agile practices across its teams, with a daily and weekly ritual of check-ins, sprint planning, and review sessions. As Alibaba embraced remote working, it also made sure its meetings were more tightly run. One person is assigned to trace time and manage the outcomes. Team members can rate a meeting’s usefulness by employing a five-star system that gives immediate feedback and positive ways forward. To address the challenge of launching a digital business with an oversized remote team, one company created a replacement workflow for product requirements that clearly outlined the use of digital tools, roles, and responsibilities as requirements moved from ideation to validation to delivery stages. Reiteration of decision-making structures like this isn’t always necessary when people can communicate directly, but their absence will be keenly felt when remote working kicks in. 
The lesson: Establishing robust working norms, workflows, and features of authority is critical, but only too easy to stint on.
The final lesson: Being able to recognize what isn’t working and changing it fast.
These are the impacts of COVID-19 on the world of work. Leadership teams that continuously learn actively identify best practices, and rapidly founded mechanisms to share ideas across the organization tend to be most successful within the future. R&D teams at one leading high-tech manufacturer created a productivity target for remote work by estimating their productivity hebdomadally relative to onsite work and identifying levers to enhance it. Within four weeks, they had progressed from 50 percent to 88 percent of their baseline. As China’s workforce begins to return to offices, these lessons from a number of its leading companies help as an example of how—with the correct structure, culture, processes, and technology—working remotely can boost productivity and morale. Employees who spend less time traveling or commuting and have a much better work-life balance are likely to be happier, more motivated, and prepared to mobilize in extreme situations. Embracing remote working allows companies to define a brand new normal that drives productivity and employee satisfaction into the longer term. Alibaba launched TaoBao, by now the world’s biggest e-commerce website, while the staff was working remotely on quarantine during the 2003 SARS outbreak. For Trip.com, an overseas working experiment in 20147 established the foundations for nice customer service and versatile working culture. Hence, bringing together all the weather can enable a brand new way of working that may make your company suited the future—whatever which will hold. These are policies that will mitigate the impacts of COVID-19 on the world of work
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
