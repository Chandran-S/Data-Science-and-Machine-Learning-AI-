import requests
from bs4 import BeautifulSoup

# Example URL
url = "https://insights.blackcoffer.com/what-if-the-creation-is-taking-over-the-creator/"

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
text = ('''What if the Creation is Taking Over the Creator?
Human minds, a fascination in itself carrying the potential of tinkering nature with the pixie dust intelligence, creating and solving the mysteries and wonders with anything but admiration. However, no matter how captivating a human mind can be, it could sometimes be appalled. It could be the hunger or maybe the desire to want more, to go beyond and unravel the limitations, or maybe something like pure greed. Humans have never stopped and always keep evolving when it comes to intelligence and this is what makes them the supreme.
Intelligence calls out for supremacy and so, what if there was to evolve something that opposed a challenge to the very human minds, to their capabilities while making them question their own importance among themselves? Artificial Intelligence came as a revolution, havoc when it first came to the light. The concept of making machines does work on their own, like granting machines –The Intelligence.
The idea of making machines work like humans came back in the 19s. Back then people didn’t believe in such a thing as making a non-living thing work, think, and carry tasks on its own, not to mention, to actually surpass humans themselves in those skills. The facts are it did. By 1997. The greatest chess player, Garry Kasparov was defeated in a chess game by a machine and this is where exactly, a top skilled human lost to a mere machine created by another who by himself could’ve never defeated him. It was a rule of power, of betterment, of skills, and the granted supremacy. Were AI and Machines just tools? Equipment?  Something that helped an unskilled person with his mind and intelligence creates something that could do the skilled work for him with perfection and precision? Well initially it was, however, as time passed as humans got drawn to the puzzle of AI, a lot changed. Human research went deeper and deeper and as a result, the machines evolved with it.
At present, AI & Machines is a growing field. As it develops and improves, it has become a part of the industrial revolution. In industries, most of the laborious work that was once taken care of by humans was now replaced by machines. Naturally, with the evolution in machines, its precision, mass productivity, quality control, time efficiencies, and all the other factors made it a better choice. A choice over humans.
This led to fear, a fear of a not-so-distant future, a future where maybe machines will be so evolved that they’ll take over the need of a human employee leading to unemployment. With the population increase around the world, it became the new tech threat for the labor market. Then again… how true is it? Does AI really oppose a threat? Will adapting to technology make millions of people lose their jobs? Will it lead to mass unemployment? Will the machines really surpass humans? Will, the creation take over the creator?
No matter how fearful the future with AI may seem, in reality, it is not that scary. Truth is AI is the present reality, it is the key that holds the power to unlock a whole next level of human evolution. Technology is growing. There was a time where technology was just an idea, but today that idea has been implemented, it’s working and is carried out. Nobody could stop the advancement and growth of Artificial Intelligence, it’s a wave that is already flowing and we as the present generation and the generations to come to have to learn, to learn to swim in this flow and avoid drowning.
Many jobs will be replaced by machines, as AI evolves it’ll keep challenging human minds and their skills. With the present COVID 19 situation, contactless cashiers to robots delivering packages have already taken over the usual routine tasks. The jobs of Secretaries, Schedulers, and book-keeper are at risk too. Manufacturing units, agriculture, food services, retail, transportation & logistic, and hospitality are all a part of the AI-affected automation. At an estimation, it is said that around 20 million jobs, especially including manufacturing will be lost to robots. As AI, robotics, 3D printing, and genetics make their way in, even the architects, medical docs, and music composers feel threatened by technology. Making us question that will AI even edge us out of our brain jobs too? Now that can be terrifying.
However, as much as machines will be replacing few jobs, they’ll also be creating new jobs.  With the economic growth, innovation, and investment around 133 million jobs are said to be generated. These newly enhanced jobs are to create benefits and amplify one’s creativity, strategy, and entrepreneurial skills. So what is the catch?
Well, it’s the skills. Even though AI is creating 3 times more jobs than it is destroying, it’s the skills that count. AI surged in new job opportunities, opportunities like Senior Data Scientist, Mobile Application Developer, and SEO specialist. These jobs were once never heard of but now with AI it’s born, however, to do these jobs or for its qualification, one needs high-level skills and to acquire those skills can be an expensive and time-consuming task. The future generation might be able to cope up with it but the real struggle is to be faced by the present two generations. It’s the vulnerability between the skill gap and unemployment and the youths are the ones to be crushed the most.
Therefore, as the advancement of AI becomes inevitable there remains no choice but to adapt, learn, equip ourselves and grow with it. The companies have to work together to build an AI-ready workplace. They should collaborate with the government, educators, and non-profit organizations and work together to bring out policies that could help understand the technologies’ impacts faster while also providing the employees some security. The economic and business planning should be made considerable for minimizing the impact on local jobs and properly maximizing the opportunities.
The employees should be provided with proper tools to carry along with the new opportunities while acquiring AI-based skills for their day-to-day work. New skills should be identified and implemented for the upskilling and continual learning initiatives. Employees will have to maximize their Robotic Quotient and learn core skills. They’ll have to adapt to new working models and understand their roles in the coming future. 
Howsoever, it’s not like AI will totally take over control, even though AI proves to be a better choice, it still has its limitations at present. First, it’s expensive, secondly, manufacturing machines in bulk is not good for the environment. Machines are also very high maintenance, therefore human labor will often come cheaper and so will be considered over machines. Underdeveloped countries will find it hard to equip their people with the upskilling and reskilling required for AI workplace and so for AI to play a role in those countries, might take years. AI can also be risky and unethical, as it’s hard to figure out who to be held responsible for in cases where an AI went wrong.
No matter, how advanced AI gets, there are some skills where humans will always have an upper hand i.e., soft skills. Skills like teamwork, communication, creativity, and critical thinking are something that AI hasn’t been able to beat us up to yet and so the value of creativity, leadership, and emotional intelligence has increased. Although, with machines coming in between humans causing the lack of human-to-human interaction, the humans seem to fade away a little.
With this era, comes the need for good leaders. Leaders who are capable of handling both machines and humans together, the ones who are organized enough to manage the skilled and the unskilled employees while providing the unskilled trainees with proper training. Leaders who hold profound soft skills and encourage teamwork while working along with machines. The ones who are patient, calm, and optimized.  
In conclusion, yes AI and machines are going to be very challenging but there’s nothing humans haven’t overcome. Adaptation and up-gradation are going to be the primary factor for survival. As we witness the onset of the 4th industrial revolution, let’s buckle up our seats and race along the highway with the essential fuels (skills) so as to not let ourselves eliminated. After all, this is an unending race with infinity as the end, all we could do is try not to run out of fuel. Try not to be outdated. ''')

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


