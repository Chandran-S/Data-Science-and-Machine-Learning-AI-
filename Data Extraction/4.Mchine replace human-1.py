
import requests
from bs4 import BeautifulSoup

# Example URL
url = "https://insights.blackcoffer.com/will-machine-replace-the-human-in-the-future-of-work/"

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
text = ('''Will Machine Replace The Human in the Future of Work?
“Anything that could give rise to smarter-than-human intelligence – in the form of Artificial Intelligence, brain-computer interfaces, or neuroscience-based human intelligence enhancement – wins hands down beyond contest as doing the most to change the world. Nothing else is even in the same league.” 
–Eliezer Yudkowsky, AI Researcher
There’s no denying robots and automation are increasingly part of our daily lives. Just look around the grocery store, or the highway, they are everywhere. This makes us wonder what if AI can replace human intelligence? What can we do to make ourselves relevant tomorrow? Let us try to find the answers to all these questions and more.
Let’s first understand what is Artificial Intelligence –
Artificial Intelligence or AI basically machines displaying intelligence. This can be seen from a machine playing chess or a robot answering questions on Facebook. Artificial Intelligence can be further broken down into many different types. There are AIs designed to do specific tasks, such as detecting a specific type of cancer. However, there are also AIs that can do multiple tasks, such as driving a car. There are many types of AIs. Among the top, most important fields are Machine Learning or ML, Neural Network, Computer Vision, and Natural Language Processing or NLP.
Machine Learning is the idea of machines being able to prove themselves similar to how a human being learns a new skill. Machine Learning also allows for the optimization of an existing skill. Machine Learning is used in many different fields and one such application is entertainment. Netflix uses Machine Learning to recommend more shows that you can watch based on the shows that you have already seen.
Neural Networks are algorithms that are modeled after the human brain. These algorithms think just like we do which can thereby give similar results to what a human being can give. Artificial Neural Networks are used in medical fields to diagnose cancers like lung cancer and prostate cancer.
Computer Vision is the idea that computers have visions. This allows them to see things the way human beings do or potentially better than human beings do, depending on the programming, camera used, etc. Computer Vision is used in autonomous vehicles for navigation from one place to another.
Natural Language Processing is the idea that computers can listen to what we say. An example of this is Siri. Siri is able to listen to our demands, process what it means, and provide you an answer based on what is researched.
Now that we know what an AI is and what it can do, Let’s talk about the issue.
AIs allow for the automation of jobs, thereby replacing what humans already do. This means more job loss and the concentration of wealth to the selected few people. This could mean a destabilization of society and social unrest. In addition to social unrest, AI improves over time. This means it becomes smarter, faster, and cheaper to implement and it will be better at doing repetitive things that humans currently do, such as preparing fast food. It is predicted that AI will improve so much over 50 to 100 years that AI will become super intelligent. This means that it will become even smarter than the most intelligent human on earth. According to many experts such as Elon Musk, this could cause the end of human civilization. AI could potentially start a war against humans, burn crops and do all sorts of tragedies once reserved for human functions. At that point, in theory, we can not stop it because AI would have already thought of all the obstacles that will prevent its goal. This means that we cannot unplug the machine, in effect AI will replace human intelligence.
But, will this happen in next 10 to 30 years?
NO! The field of Artificial Intelligence is sophisticated enough to do many human tasks that humans currently do. Currently, AI is not smart enough to be empathetic to humans and cannot think strategically enough to solve complex problems. AI solutions can be expensive and have to go through many different tests and standards to implement. It also takes time for AI to improve. For example, Boston Dynamics, one of the world’s top robotics company had a robot in 2009 that needed assistance to walk. Fast forward to 2019, not only the robot could walk by itself but it could jump over objects, do backflips and so much more. In addition to the timing, it takes time for the price of any new technological solution to drop to a point where it is affordable. For example, a desktop computer costs around $1000 in 1999 but now you can get a significantly more powerful laptop for the exact same price. AI will go through the same curve.
But what happens after those 10 to 30 years? Will AI make human intelligence obsolete? Maybe. As we have proven earlier AI will become faster better and cheaper. As this happens, more and more companies will use AI technology to automate more and more jobs to save money, increase productivity, and most importantly, stay competitive. As we have demonstrated, AI will become better through repetition via the use of machine learning. The only difference is that AI will be able to learn faster as time progresses due to the amount of data that is available today. It will also be able to learn from other machines or similar machines to learn how to optimize its tasks or new important skills. However, AI also just not do repetitive and routine tasks better, it will also be able to understand emotional intelligence, ethics, and creativity. This seen in three distinct example- IBM
IBM uses its IBM Watson to program the AI to create a movie trailer. Fox approached IBM and said they have a movie coming out on AI #Scifi horror. They asked IBM if their platform IBM Watson could a trailer by reviewing and watching the footage and searching for scary,
Sad or happy or other moments in the movie that provoked quality emotions based on how the machine was programmed to identify such emotions in a quantifiable manner. IBM Watson was able to generate a trailer for the movie Morgan. The result, a movie trailer created by machines example – Google
IN 2018 google demonstrated an AI assistance that could take calls and do simple stuff. The AI was able to set up an appointment! What was more fascinating was that it was able to understand the nuances of the conversation. The receptionist thought it was a human being that was calling her. That is a very primitive version of what is possible with this technology. Eventually, it will be able to have conversations just as human beings do, making many sales jobs obsolete. example – AI generated art
In 2018, a Paris art collective consisting of three students used artificial intelligence to generate a portrait. It generated the portrait painting by studying a set of fifteen thousand art images on wiki art. It was estimated to be worth between seven thousand to ten thousand dollars. The painting sold at an auction for four thirty-five thousand US dollars.
However, we cannot for sure say that AI will replace human intelligence. This is because we as a society have started asking hard questions and questioning ethics. Elon Musk founded Open AI, a research lab whose whole purpose is to promote and discover artificial intelligence in a way to benefits humanity. In addition to this, there are many factors that affect the long-term outcome of AI replacing human intelligence. Like, to what degree will other humans allow for AI to take over? Depending on the field, do people even want Artificial Intelligence to help them? Or will they prefer a human counterpart? While we may not be able to control what happens in the long run, we can definitely secure our short-term future.
Strategic and creative thinking
The ability to think outside the box is very human. There are thousands upon thousands of slightly different possible outcomes that may result from every distinguishable action that the human mind with its ability to judge from experience is programmed for these purposes in a far more sophisticated manner than AI can currently achieve. As the billionaire founder of Alibaba, Jack Ma famously said – “AI has logic, human beings have wisdom”.
Conflict resolution and negotiations
With our understanding of the complexities of human-related processes and our ability to improvise and judge, we are far better equipped to deal with conflicts than robots are ever likely to be.
AI may be able to recognize faces and images but it can rarely successfully read the feelings of those faces. Humans, to lesser or greater degrees, are capable of an accurate analysis of emotional subtext. With the application of intuition and the use of delicately worded or elusive languages, through these methods, we are able to properly judge how a person feels.
Interpretation of Gray Areas
Robots and computers function well when presented with quantifiable data. However, once the situation enters a gray area, whether this term refers to morals, processes, or definitions robots are more likely to falter.
Critical thinking
Humans are capable of responding to more indicators of quality than computers are. While an AI system may be able to analyze documents according to the true or false statements made within the text, we can judge whether or not it is well written and analyze the implication of the use of certain words and the overall meaning of the content.
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

