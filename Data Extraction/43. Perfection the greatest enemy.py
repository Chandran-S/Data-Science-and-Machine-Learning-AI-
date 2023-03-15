
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/is-perfection-the-greatest-enemy-of-productivity/'

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
text = ('''How to Overcome Your Fear of Making Mistakes
No one can reduce mistakes to zero, but you can learn to harness your drive to prevent them and channel it into better decision making. Use these tips to become a more effective worrier.
Don’t be afraid or ashamed of your fear. 
Our culture glorifies fearlessness. The traditional image of a leader is one who is smart, tough, and unafraid. But fear, like any emotion, has an evolutionary purpose and upside. Your concern about making mistakes is there to remind you that we’re in a challenging situation. A cautious leader has value. This is especially true in times like these. So don’t get caught up in ruminating: “I shouldn’t be so fearful.”
Don’t be ashamed or afraid of your fear of making mistakes and don’t interpret it as evidence that you’re an indecisive leader, or not bold, not visionary. If you have a natural tendency to be prevention-focused, channel it to be bold and visionary! (If you struggle to believe this, identify leaders who have done just that by figuring out how to prevent disasters.)
Use emotional agility skills. 
Fear of mistakes can paralyze people. Emotional agility skills are an antidote to this paralysis. This process starts with labeling your thoughts and feelings, such as “I feel anxious I’m not going to be able to control my customers enough to keep my staff safe.” Stating your fears out loud helps diffuse them. It’s like turning the light on in a dark room. Next comes accepting reality. For example, “I understand that people will not always behave in ideal ways.” List off every truth you need to accept. Then comes acting your values. Let’s say one of your highest values is conscientiousness. How might that value apply in this situation? For example, it might involve making sure your employees all have masks that fit them well or feel comfortable airing any grievances they have. Identify your five most important values related to decision-making in a crisis. Then ask yourself how each of those is relevant to the important choices you face.
Repeat this process for each of your fears. It will help you tolerate the fact that we sometimes need to act when the course of action isn’t clear and avoid the common anxiety trap whereby people try to reduce uncertainty to zero.
Focus on your processes. 
Worrying can help you make better decisions if you do it effectively. Most people don’t. When you worry, it should be solutions-focused, not just perseverating on the presence of a threat. Direct your worry towards behaviors that will realistically reduce the chances of failure.
We can control systems, not outcomes. What are your systems and processes for avoiding making mistakes? Direct your worries into answering questions like these: Is the data you’re relying on reliable? What are the limitations of it? How do your systems help prevent groupthink? What procedures do you have in place to help you see your blind spots? How do you ensure that you hear valuable perspectives from underrepresented stakeholders? What are your processes for being alerted to a problem quickly and rectifying it if a decision has unexpected consequences?
Broaden your thinking. 
When we’re scared of making a mistake, our thinking can narrow around that particular scenario. Imagine you’re out walking at night. You’re worried about tripping, so you keep looking down at your feet. Next thing you know you’ve walked into a lamp post. Or, imagine the person who is scared of flying. They drive everywhere, even though driving is objectively more dangerous. When you open the aperture, it can help you see your greatest fears in the broader context of all the other threats out there. This can help you get a better perspective on what you fear the most.
It might seem illogical that you could reduce your fear of making a mistake by thinking about other negative outcomes. But this strategy can help kick you into problem-solving mode and lessen the mental grip a particular fear has on you. A leader might be so highly focused on minimizing or optimizing for one particular thing, they don’t realize that other people care most about something else. Find out what other people’s priorities are.
Recognize the value of leisure.
Fear grabs us. It makes it difficult to direct our attention away. This is how it is designed to work so that we don’t ignore threats. Some people react to fear with extreme hypervigilance. They want to be on guard, at their command post, at all times. This might manifest as behavior like staying up all night to work.
That type of adrenalin-fueled behavior can have short-term value, but it can also be myopic. A different approach can be more useful for bigger picture thinking. We need leisure (and sleep!) to step back, integrate the threads of our thinking, see blindspots, and think creatively. Get some silent time. Although much maligned, a game of golf might be exactly what you need to think about tough problems holistically.
Detach from judgment-clouding noise. 
As mentioned, when people are fearful they can go into always-on monitoring mode. You may have the urge to constantly look at what everyone else is doing, to always be on social media, or check data too frequently. This can result in information overload. Your mind can become so overwhelmed that you start to feel cloudy or shut down. Recognize if you’re doing this and limit over-monitoring or over checking. Avoid panicked, frenzied behavior.
On its own, being afraid of making mistakes doesn’t make you more or less likely to make good decisions. If you worry excessively in a way that focuses only on how bad the experience of stress and uncertainty feels, you might make do or say the wrong things. However, if you understand how anxiety works at a cognitive level, you can use it to motivate careful but bold and well-reasoned choices.
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

