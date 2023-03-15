
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/will-we-ever-colonize-outer-space/'

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
text = ('''Will we ever colonize outer space?
Nature has blessed humanity with earth, a rounded globe where all essential necessity lies in the hand of a human. Though in this 21st-century building Space shuttle and exploring space has become an obvious activity among the human raise, now with the inclusion of Start-ups like Space X once dreamed for reaching out at space now is coming true. We all have to agree upon a fact that in the Growth of industrialization we have generated a humungous carbon footprint and colonize at outer space will create a much-needed area for our survival.
With time the human race has visualized and has undergone many transformations, and these transformations may for starters were not that effective but were a major role at existence to this age. Be it a natural transformation like shifting of tectonic plates or manmade like Dumping water and damaging water bodies, somehow have affected usual life of humans so creating space of humans is what next humans are eyeing for.
All this idea for Colonizing in outer space emerged when NASA was able to launch a space shuttle with a heavy payload in the late 1990s this idea didn’t stir brain at that time but later was like apple eyes when various articles from astronauts and their writeup came into limelight and this idea saw its progression, space exploration was not a new normal, companies were petrified by the number of investments that these kinds of projects ask for. But things change with the entry of PayPal’s owner Elon Musk, in 2002 where he laid stones of Space X, with NASA providing the launchpad for their exploration.
Being space travel legitimate there where many bottlenecks knowing the nature of outer space will boost the idea of colonizing, here one has to realize the optimum. But the main target that these company put forward with themself was to not just get into other planet but to get back at earth safely, earlier these projects due to high fund was always subdued and priority was given to the rocket technology. Thus, this unimaginable quest for the search of space was far-fetched though.
Now being in 2020 we do not find it out of a place of colonizing outside of space, the kind of emergence of technologies has led to the escalation of spreading the roots of human existence, the plan for reaching out to outer space too plays a key role for this idea too. Proper Planning, Analysing, Organising, and Executing were required for these projects to be successful for deeper know how I would like to mention here an example of Space X, According to them they wud be successful for sending Rockets by 2022 and would able to colonize by 2050, For achieving these vision they have planned a total of 36 times pre-launch of the same rocket which they will be launching by 2020. Not only that they have already placed their star link satellites which will provide the basic network connectivity for their colonized world, they have planned this to that extent that they have targeted a profit of $ 22 billion yearly by 2025. Already they have sent the first manned rocket successfully what Elon musk plan is to create that infrastructure by which going to Mars with cost as a regular air flight ticket all this will lead to better chances for colonizing at another planet. With setting up their commercial flight once started to the red planet it wud just be a stepping stone for the colonizing at the outer space.
But the fundamental change that people have to undergo while living on another planet will be a change in gravity and being able to be near to live support if any technical issues arise to their suit until unless we find a similar place like the earth that is the change which humans have to adjust. Aspects like Ultraviolet rays too will play an immense role in the survival, but with the rate of technology expanding and challenging at very boundaries there are not many days left when we find solutions to these aspects but this idea of colonizing has more to do with monetary terms it has been projected that building a colony at space will cost at about $10 Trillian which is more than many countries GDP. Placing that much of an amount and getting funding for such highly anticipated projects is tough. Elon to have addressed with these questions with an open Door he believes that this cost is nothing when we find a route to colonize space and these costs will be justifiable when space exploration will reach new heights.
Also, our world is around water and the percentage of land comparison to water is too less. Whereas the population is increasing with the speed and the level of excitement in order to search for new things and the obsession with the new and innovative things never going to finish. Because in the end, we do not satisfy what we have achieved, we are only looking for other things and crave to achieve that.
In order to achieve that planning is most crucial stage. As of now we have searched 9 planets but still didn’t shortlisted the outcomes. But according to the 3-decade theory of ELON MUSK, it seems to be possible.
As the most important thing to survive is the atmospheric environment, whether is this suitable or not, and in order to make it suitable what steps had to take and maintain the suitable level of oxygen.  The missions are now what everyone is focusing about. Finding the source of water, because its 2nd crucial thing which is needed to survive.
Apart from the excitement and the things, the threats or the uncertainties are bound to happen like any kind of mammal which could occur during construction or after that. So the security for that or any protection so that uncertainties could be replicated.
After the basic things, other factors like what are the natural habitats available there, whether the plants could be grown over there or not. If yes, then what types of plants could survive there.
The environment for the animals is safe or not. Because to balance the nature we need all types of animals, natural habitat, even insects and other small things which seems to be ignorable but plays an important role in maintaining the environmental balance. For e.g.  – Ants are among the leading predators of other insects, helping to keep pest populations low.  Ants move approximately the same amount of soil as earthworms, loosening the soil in the process and increasing air and water movement into the ground.  They keep the ecosystem clean of dead insect carcasses and aid in the destruction and decomposition of plant and animal matter. So, every small thing contributes in its own way.
Another matter of concern is the level of artificial intelligence and machine learning required to maintain that kind of requirement. Because as the current scenario, we need energy to run any vehicle or even for the robotics. So, the availability of the energy and the source of energy is needed.
And apart from this, the heat of sun is equally important in order to maintain the level. Because with artificial air we can survive but not for so long. So, in order to remain healthy as well the calculation has to be done. When thinking about disease, you may have heard in the news about the concern that antibiotics (such as penicillin) – which help fight infections – may eventually stop working. Global health organisations are trying to reduce the use of antibiotics, especially for conditions that aren’t serious. This is because their overuse in recent years means that they’re becoming less effective.
But still we can survive without antibiotics. Because there are ways of dealing with diseases caused by bacteria such as isolation which is how they were treated before penicillin.
It might sound like something set firmly in the realm of fantasy, but experts in private industry and governments around the world are trying to understand how feasible it would be to establish a lunar base. Some scientists think humans could survive comfortably on the moon. In some ways, the very minimal gravity of the moon might actually be more conducive to life than the microgravity astronauts experience on the International Space Station.
Although it hasn’t been formally tested, some experts hypothesize that the small amount of gravitational force put on an astronaut’s body when on the moon could help stem some of the adverse effects like bone-density and muscle loss that space flyers experience while living in microgravity on the International Space Station.
By using the moon’s indigenous material, space agencies can save money on the cost of flying pricey missions to and from the moon’s surface. Once on the moon, instead of having to stage costly missions aimed at delivering oxygen and other necessary resources from Earth, experts might be able to actually use the material to manufacture gasses needed to sustain life on the satellite.
On earth, we are protected from radioactive solar winds and cosmic rays by our own magnet field known as the magnetosphere. But out in the space, we can’t rely on this type of protection, and in order to make that protection, it could take a lot but yes with more upgraded technology it could achieve.
Although till now the approximate value of the project is $10 trillion. But the aim is to make the fare equivalent to the normal air fare.
As once we are comfortable with all the resources, securities for uncertainties, risk analysis, and the probability of survival of the human being and many other factors after consideration, then development won’t take much time.
Because to construct a building it take normally 8-9 months. Which is totally acceptable.
But the main concern is the density of the land, i.e. whether the land has the capacity to bear that much weight or not. And the type of land is adaptable for cultivation or nor. Although land can be converted into different types of cultivation but with the help of upgraded technology.
In order to transport the equipments on the regular basis, a large amount of fuel Is required, and other factors as well and Built the propellor depo for rocket landing building various infrastructure for the city one by one will create a way forward for idea of Colonizing at outer space.
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

