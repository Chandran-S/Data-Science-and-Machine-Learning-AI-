
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/what-is-the-chance-homo-sapiens-will-survive-for-the-next-500-years/'
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
text = ('''What is the chance Homo sapiens will survive for the next 500 years?
We’ve really done it this year. Like an insatiable glutton, the law of averages has come home to roost. We should’ve taken the hint when on the 1st of January, 66 people lost their lives in the Jakarta floods. What followed was like the highlights reel of a disaster movie franchise – a volcanic eruption in The Philippines, irrepressible bushfires in Australia, earthquakes in Russia, Iran, Turkey, India, and China. And speaking of China.
2020 has brought home the fragile mortality of the human race into sharp focus. As global Covid-19 deaths stoutly push past the grim 1 million marks, we have no choice but to question our place in the universe – are we the all-conquering masters of our domain, or mere tourists in a ruthlessly apathetic ecosystem? Is the human race on the ubiquitous three-part literary arc that defines every story, every life, every civilization – ascent, apex, and descent? Maybe when Michael Jackson unveiled his moonwalk in 1983, or when Barack Obama stepped into the White House as President of the United States in 2008, or indeed when MS Dhoni lifted the Cricket World Cup in 2011, we peaked, as a species, and everything since then has been a steady unraveling.
500 years is a long time. For context, the world population in 1500 AD was a mere 461 million. The 16-fold explosion since then is unprecedented in history, but we might just be at the tip of an iceberg. Though fertility rates are dropping and more and more people are foregoing the chance to have babies, we might just have crossed the threshold – the population projections for the year 2050 is 9.8 billion, and for 2100 is a whopping 11.2 billion1. Somewhere out there, Malthus is cackling in his grave. The year 2500 suddenly seems a long way off, and this conversation seems ever-more pertinent today. The Bulletin of Atomic Scientists is not optimistic – the famed Doomsday Clock they maintain is the closest to ‘midnight’ (our proximity to global catastrophe), since its inception in 1947. Global warming? Check. The threat of nuclear war? Check. Ongoing pandemic? Check, check, check.
And yet, hope floats, for three reasons. Mankind may just have its back to the wall right now, but there are three shoots of potential that might just help us make it to 2500 AD – the advent of a basket of disruptive technologies (artificial intelligence, bio-enhancement, genetic engineering), the private sector focus on space exploration and terraforming, and good old fashioned human resilience. While the first two factors will no doubt be critical to human survival, it is the third one that we must pin our hopes on – our long-demonstrated history of surviving whatever nature, the universe, or our own self-destructive tendency, throws our way.
The Next Superman?
In the 13th century, in the Italian town of Pisa, an enterprising tinkerer developed the first eyeglasses, for a local friar with weakening eyesight2. All of a sudden, there existed an external device that could amplify our senses, a tool that gave us an advantage in survival. Today, LASIK surgeries obviate the need for eyeglasses entirely. Hearing aids give the gift of auditory perception back to those who had gotten used to a world of muffled voices and unheard sounds. The iPhone routinely comes with an augmented reality tool that allows us to measure the length of objects in front of us.
But the real science starts where the imagination ends – augmented reality glasses, smart wearables, and virtual reality tools will be ubiquitous in the next few decades. But what after that? Science has the answer, and it’s both thrilling and scary. Artificial intelligence has become the stuff of fable, the filler for all questions left unanswered. But AI, combined with bio-enhancement and genetic engineering, might just lead to the evolution of what some are calling Homo nouveau3. Homo nouveau will be smarter, faster, more agile, and better equipped to adapt to what promises to be a world that is VUCA beyond our imaginations. What might such a human being look like?
They might have a small chip embedded in their brain that utilizes artificial intelligence for enhanced sensory perception. What does that mean? It means that they would be able to see better, focuses their attention for longer, hears what they want to hear, and communicate the appropriate reaction to the rest of the body. Through a chemical in the blood, this AI chip would be able to demand the appropriate response from the body. Bio-augmentation of limbs and organs, internal and external, would mean that what a person can or cannot do is no longer determined at birth, but can simply be bought. All of a sudden, the average Joe can run faster than Usain Bolt, swim better than Michael Phelps, and…fly? Maybe. Genetic engineering will be the missing link. Already there are feverish conversations about a dystopian future featuring designer babies and a digital divide that simply cannot be overcome because it is inbuilt into one’s DNA. The breakthrough with CRISPR-Cas9 might just be the key to unlock the mysteries of DNA manipulation. So what if there’s no food left? Our body AI will adjust our appetite accordingly. No water? Absorb humidity from the air through specialized pores in the skin. The human being in 2500 AD may not be how we recognize one today. That may be our only shot.
Galactic Dominance
SpaceX, led by its mercurial leader Elon Musk, has been the leader here. The SpaceX Mars Programme is based on a very simple premise – as the earth’s closest planet in terms of distance and terrestrial conditions, Mars would be our best bet for colonization. Musk has invested billions of dollars in the Mars Space Programme and remains a fervent believer in the concept. And among tech visionaries, Musk is not alone. Jeff Bezos, the richest man in the world, owns Blue Origin, which simply aims to make spaceflight cheaper through incremental technological growth. Bezos, who took an online retailer of books and turned it into an ever-expanding behemoth, is not a man who thinks small. With more and more business leaders finding spaceflight and planetary colonization a tantalizing prospect, there will inevitably be a concerted push to developing an actual colony on another planet. And when the pull-factor to this development starts hitting diminishing returns, there will be the inevitable push factor as global warming and the possible increase in the eruption of pandemics begin to take their toll. It might just become a more feasible option for people to find an alternate home, if not on Mars, then on one of Saturn’s moons.
A human colony on Mars sounds like a concept straight out of science fiction, but so did a permanent station in Antarctica until a few decades ago. Mount Everest seemed like an unapproachable summit until someone went ahead and planted a flag at the peak. Today, hundreds of people every year attempt the climb. As spaceflight becomes more reasonable, as the urge to explore supersedes the inertia of investment, we become closer to our best chance of survival – leaving planet earth behind and finding another home, perhaps one more forgiving of our follies.
The Human Spirit
The introduction to this article includes an illustrative list of disasters and misfortunes that have struck the global community this year. And yet, we survive. Businesses surge ahead, people adapt to a world of masks and social distancing, the world goes on. Earthquakes, tsunamis, pandemics, we’ve seen them all before and survived, and the human race, in its intrepid exploration of the world, pushes onward and upward. Technological visionaries envision a future that the rest of us cannot see, and then invest money in their vision. Gradually, what seemed unthinkable suddenly becomes real – the moon landing is the best possible example of that. Google launched Google Glass as the first augmented reality wearable technology, and suddenly we could foresee a future with smart eyewear. While the product didn’t quite catch on, that doesn’t mean other companies aren’t trying. Bio-augmentation is already a fast-developing industry, while CRISPR-Cas9, the “genetic scissors” is being held back only by regulatory bottlenecks. How long before the prospect of fiddling with the gene code becomes not a luxury but a necessity? The human spirit, the tendency to survive and thrive at all costs, will eventually win.
The human race will survive to 2500, of that there is no doubt. The real question is, will the person who exists in 2500, with bionic chips and a bio-enhanced body structure and a modified genetic code, be called a Homo sapien? Or is Homo nouveau the way forward?
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

