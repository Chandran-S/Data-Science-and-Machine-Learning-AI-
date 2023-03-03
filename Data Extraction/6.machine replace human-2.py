
import requests
from bs4 import BeautifulSoup

# Example URL
url = "https://insights.blackcoffer.com/man-and-machines-together-machines-are-more-diligent-than-humans-blackcoffe/"

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
text = ('''Will machine replace the human in the future of work?
Where is this disruptive technology taking us? Take it or leave it, disruptive technology always creates new jobs much more than depleted jobs. You might notice certain jobs disappearing but those jobs are the jobs that transform humans to robots, to machines, and the technology is creating machines to replace them.  Technology creates the data analysis tools to manipulate and create custom scenarios using artificial intelligence (AI), Big Data and Machine Learning (ML) algorithms to predict and drive consumer behavior. Data Analytics tools, such as Google Analytics , and others are available today for free, and, if used correctly, can help organizations save millions, maybe billions of dollars of sales and marketing.
Before I go on, I think it’s best to level set on what constitutes machines. In the context of this article , machines describe computers and computerized equipment, like robots, that have been programmed to learn, sometimes like humans. Occasionally we call this Artificial Intelligence (AI), other times we call this machine learning, and still other times we call this robotics. And yes, these are technically different things. These bots are more efficient than humans in some specific domains and are growing smarter with each passing day. They can do some really tough tasks which are considered difficult for any human being.  But, within the broad discussion related to the future of work, these are totally interrelated. Factory floors deploy robots that are increasingly driven by machine learning algorithms such that they can adjust to people working alongside them. A machine can work efficiently only it has abundant data and information about the work which is being imparted daily to them. But with every forward step & advancement in technology, a threat is proliferating, a threat of being replaced on our work front. Every passing day is sealing some jobs for humans all over the globe. Similarly, AI is being used to turn hand-drawn sketches (done by humans) into digital source code.
Companies are clearly developing their AI and robotics expertise with the idea that through these technological innovations they’ll be able to
Of course, it’s not just machines and creatives working together either. In another example, Amazon has employed more than 100,000 robots in its warehouses to efficiently move things around while it has increased its warehouse workforce by more than 80,000. Humans, in Amazon’s case, do the picking and packing of goods while robots move orders around the giant warehouses, essentially cutting “down on the walking required of workers, making Amazon pickers more efficient and less tired.” Plus, the robots “allow Amazon to pack shelves together like cars in rush-hour traffic because they no longer need aisle space for humans. The greater density of shelf space means more inventory under one roof, which means better selection for customers.”
During the next few decades (or maybe sooner), the notion of work and whether it is handled by a human or a virtual being will hinge on predictability. As they are starting to do today, machines will manage the routine while humans take on the unpredictable – tasks that require creativity, problem-solving, and flexibility. In this context, robotics should be seen not only as a means to improve operations efficiency but also to improve the quality of life for workers.
Although it is obvious that human factors involved in a work activity impact job automation, it is also true that highly repetitive tasks—and even mechanical ones—are ideal for robots. Besides greater efficiency and speed, automation leads to a lower risk of accidents, greater control and autonomy, and above all, fewer costs for organizations.
Although artificial intelligence and machine learning make us believe that robots are endowed with superior intelligence, in fact they don’t yet have the ability to learn from experience and to respond to unknowns. So as things stand, however much processing speed and automatic learning a robot has, it doesn’t beat factors innate to the human brain. Humans are still a very essential part of the process. Think about delivering services to a client. Most customer challenges are routine, but humans play a very important role in addressing new issues, solving them the first time they appear, and then consolidating the process into the system.
While machines and humans are placed in proximity,  robots can be expensive, but this doesn’t apply to all types, especially those based on Robotic Process Automation (RPA), where the development process incorporates algorithms that significantly reduce costs.
Moreover, think of how domestic robots—be it a vacuum cleaner, a lawn mower or a pool cleaner—are increasingly part of our daily lives. This level of consumption that robotics has attained makes it affordable to automate tasks in modern homes to obtain greater control, security and comfort.
The division between humans and machines has been clear – I’m here, the machine is there – but that boundary is getting fuzzier. Smart prosthetics fuse seamlessly with our bodies, making up for lost limbs or providing additional strength, stability, or resilience, as seen in exoskeletons donned by assembly line workers.
We use our smartphones symbiotically, but what if they were integrated directly into our bodies? Think a smartphone in the form of a contact lens capable of transparently delivering augmented reality images straight to the brain. Think it sounds like science fiction? Think again. The first prototypes have already been built.
Soon, brain-computer interfaces could become seamless as well, creating a new synergistic relationship between the cloud and us. At that point, the question of who knows what would be moot; you ask me a question and I know the answer. Sometimes that answer will be stored in my own neural circuitry, but most of the time it would come from the connection of my neurons to the web. Our brain’s decision process is influenced by the way it has been “educated” by the cultural context. These external factors are influencing our decision processes to the point that in certain situations, we can legitimately claim that influence has been so strong that our brains can’t be held accountable for the choices made. The point I’m trying to make is that we humans are in symbiosis with our cultural environment and the tools – both physical and conceptual – that we have been taught to use. My guess is that the transformation will be subtle.
Practically speaking, robots growing to the point that they take over the world and then start creating smarter, better robots are impractical and should not even be a concern. None of this is expected in the near future, not by a long shot. If you’ve been to an ATM, waited for a PC to boot up after a catastrophic failure, or had a game crash on your X box just when you were about to reach a checkpoint, you understand that we are not in a world where machines do everything perfectly right. Before they can take over all of our jobs, they need to be able to do theirs’ flawlessly; until then, we can depend on humans to mess up our lives. 
This isn’t a win-or-lose situation. We’re going to wind up as a partner to our smarter machines, and that partnership will be fostered by our augmentation through technology. Machines will play an essential role in this augmentation and, as with any successful technology, they will fall below our level of perception. In the end, the revolution will be silent and invisible.
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

