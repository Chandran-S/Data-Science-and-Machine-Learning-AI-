
import requests
from bs4 import BeautifulSoup

# Example URL
url = "https://insights.blackcoffer.com/how-ai-will-change-the-world-blackcoffer/"
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
text =('How AI will change the World?'
'The way work is being done now is destined to undergo massive transformational changes which will impact humans and their ways of working dramatically. With the development of the new machine programs, A.I. is all set to take over the humans in their workplace as no other did. Now we are not only in competition with other beings but with robots too. And robots will overcome us in our fields of work.'
'At present we are being surrounded by A.I. from dusk till dawn, from facial recognition present in our mobile application to dating websites/applications which uses decision making as their algorithms and learn from the past data as well. It is believed that A.I. has grown over 270% over the last years.'
'First, let us know what A.I. is and all the fuss going on about it?A.I. as defined by the internet is simply ‘simulation of human intelligence in machines that are programmed to think like humans and mimic their actions’. This translates to, it can work as a human being just at fast speed and with 100% accuracy.In my definition I would define A.I. as the god form of human beings.'
'Now the question that arises is, what does A.I. entails for the future?'
'A.I. is fancy enough to continue existing in our minds all the time, but it does come with certain limitations and threats which is a cause for a peaceful sleep for most of the workers. With its introduction to different areas of the workplace, it is clear that half of the human jobs would be taken up by A.I. and then there would arise a need for more jobs for humans in new upcoming AI-based ventures in different industries. Thereby giving us our fair share back to us.'
'In a survey, it was found that some believed that AI will be devastating for humans and while some professionals and tech-savvy people believed that inculcating AI technology into business and our daily lives would be a remarkable step as it will lead to the flourishment of business in the future and give them a competitive edge over their rivals.'
'People believed that when these advanced technologies would come together to work with humans, they will produce a smarter strategic decision with productive collaborative practices. More modern technology prevailing in the organization will lead to stress reduction and produce more satisfying results thereby making the organization more efficient.'
'The Organizations who are in use of the AI technology responds by saying that their managers are more comfortable using A.I. and are accustomed to it and the organizations are now looking forward to having integration of higher-end technology systems, as it is believed that new technology will result in more productivity thereby making more profits in the long run.'
'Many jobs today require AI and humans working in collaboration, which creates a positive signal that in the future to humans would be working closely with the technology. Beyond just training and developing these machines, humans would be working in close vicinity with them and making decisions on how to act on the result that is given by the machine.'
'It can be said that both the AI advanced technology and human can’t remain in the workforce without each other, as the technology will produce accurate and top-notch results but it requires someone to make delivery.e.g., in a firm, an AI-based system produces results based on historical data but there is a need for someone to analyze and communicate and present this data to the respective stakeholders whether inside or outside of the organization.'
'Various uses of artificial intelligence technology in day-to-day functions in regards to interactions with humans are:'
'Artificial intelligence as we know works on algorithms, neural networks, and deep learning which all are analytical tools that help AI in taking analytical decisions based on the data provided.'
'Whereas, humans on the other hand take higher-level decisions based on ‘Intuition’ sometimes, which refers to the gut feeling that generates in humans concerning any situation or challenge.'
'AI alone can’t work to handle critical situations on its own as it needs humans for it to reciprocate them and share the information with stakeholders, and humans alone can’t anticipate much on the accurate and fast-paced analytical solutions to the problems persisting in place of the situation.'
'The strength of humans and AI working in synergy can be surprisingly beneficial and advantageous to organizations.'
'It is believed that machines in the future would be eating up our traditional jobs. But the reality seems to be turning otherwise.'
'The future trend shows that in future the AI-powered technology would take up jobs that were being done by humans but in return would produce more jobs that would require human interference with them. As and when the newer technology is approaching more and more countries are now proceeding towards GIG Economies, and so in the future, we can witness an increase in freelance jobs and the permanent labor market norms could reduce drastically.'
'Some of the jobs today will be replaced by AI which is in the transportation or retail commerce sector that can be 100% automated in the future years. There is an ‘Amazon Go’ store that uses this technology which goes by the name ‘Just walk out technology’ wherein there is no need for any human-induced workforce and all the operations are carried out by AI-powered technologies, which is indeed a breakthrough technology in today’s world.'
'Rather than eating up our jobs AI in return will be creating more jobs in the future by creating massive innovations thereby fueling up many new industries and thereby giving us our fair share of jobs back.'
'There will be a lot of demand placed on the upcoming young workforce which is also categorized as ‘Gen Z’. They are expected to know more about technology and would be high in demand. These young generations are required to learn new skills which are needed to survive in the dynamic changing environment, and as most of the activities that are carried out by workers will be automated, there will be demand for people working in the back office and maintaining and developing the technology to its best versions.'
'AI powered technology has its limitations which makes it a rigid system to hold on to and also which makes it a costly affair at the initial stage.'
'It was predicted that the cost of electricity to power a supercharged AI model was around $4.6 million. So, this super-powered AI can be purchased only by big fortune firms and thereby creating more value to their net worth.'
'One of the major limitations of AI is that it can contain biased data as the scientists who put in the data can create biasedness and so the resultant output of the same would have a biased report.'
'These machines as do not have neuroscience-based technology in them yet which enables them to carry human emotions to understand complex situations and a creative way out of that, they tend to have a lack of out of box thinking which in the case is rigid in themselves as they are programmed to work on a single task and they cannot perform more than a single task at the same time.'
'It is also believed that there exists no creativity among the computer, no matter it is fast-paced, but they are not intelligent.'
'Businesses and organizations need to understand and anticipate the opportunities that the future holds for them and they need to start training their employees based on today’s dynamic changing technology. While it’s still unclear what does the future holds for us, but the anticipation of it could benefit us in several ways.'
'As we are unclear about what the future looks like, we need to think in probable terms how it could turn out to be and then employ specific training programs for the employees of the organization.The training the employees are needed to be done on a continuous and lifetime basis which means that education won’t be only limited to PG degrees but will be now a lifetime process of learning.'
'As the Covid-19 changed the scenario of the work patterns around the world, we now need to think strategically about the working dynamics of the future and how does it look like compared to the pre-covid and post-covid scenario.'
'Employees will be playing a major role in transforming the organizations and work practices in the upcoming future, so organizations need to select and recruit the best candidates among the pool and then provide them with best practices of the new machines and make proficient in their area of work. Policies need to be developed to hire the best people and then retaining the talent in the organization.'
'There needs to be a continuous scanning of the environment by the organizations to comprehend any new trends and assess them, not all trends will be beneficial for organizations, they must be aware of the prospects and plan for the future systematically and consistently.'
'There lies a possibility in future certain years from now, we could have machines who will have general human intelligence who would be able to answer deep meaningful questions asking ‘Why are the curtains blue?’, would be able to clean cars, play politics and tell jokes to us, and by using deep and machine learning programs their level of intelligence would be beyond mathematical calculations to us. That’s how good machines will be in the future, but to make ourselves competitive with machines, we would need to train ourselves for the impending ambiguous future ahead of us.'
'Humans and machines need to work in synergy to get beneficial and satisfying results for both parties.'
'Machines are indeed going take away many of our jobs, but let me make you sleep peacefully tonight, the machines aren’t arriving until we’re retired.')

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

