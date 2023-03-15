
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/lessons-from-the-past-some-key-learnings-relevant-to-the-coronavirus-crisis-2/'
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
text = ('''Lessons from the past: Some key learnings relevant to the coronavirus crisis
“The more you know about the past, the better prepared you are for the future.”
As we speak, the world finds itself engulfed in one of its worst crises in recent times. The global COVID-19 pandemic has caused never-seen-before disruption in both public and economic life. Not only have factories shut down or supply chains abruptly stopped or millions of workers stranded, but festivals suspended, families separated across countries and public-healthcare systems put under tremendous stress. Such a scenario is a first-time for many, but the world has been through pandemics of a similar scale and nature before, and after the due struggle, emerged victoriously. As the more experienced amongst us would remember, the SARS and MERS outbreaks in the early 2000s presented a similar scenario, although at a much lower rate of infection, yet with higher rates of morbidity. Going back further in time, the 1918 Spanish Flu brought a world just out of the first world war, to a standstill. Considered to be the worst amongst modern-era pandemics, the Influenza pandemic affected one-third of the world’s population and 50 million lives. Yet, preventive measures such as social-distancing, quarantine, mass-vaccinations, public support, and restraint have proved to be successful in all such situations, even being effective in occasional plague outbreaks across the world. We face a similar situation today – an extremely infectious virus, which has already spread globally, confined people to their homes and is causing tremendous economic loss every minute economic activity remains suspended, affecting both the government and the industries alike. 
Can past similar incidents guide industries in such trying times? Can our governments take note of how public-policy measures adopted during the Spanish flu, SARS and MERS outbreak helped in eliminating such pandemics?
What can the Government learn?
The panic due to the outbreak of COVID-19, and more importantly, the lockdown measures to deal with it, are quite similar to those during the 1918 Influenza Pandemic, or as it is more commonly referred to as, the Spanish Flu. Large cities around the world were put under strict lockdowns, and businesses worldwide came to a grinding halt as large chunks of the population became bedridden.
The immediate consequences were the same as the ones observed in the past few weeks – rampant unemployment, supply shortages, and heavy reliance on social security systems, causing a sudden strain on national economic resources.
Impact & Learnings
Effective and transparent communications are one of the most crucial and useful tools in disease control. Iran presents a great example of the damage that can be unleashed by a media-blackout. Secondly, censorship might not be the best way to appease the masses during such times. The long-term political impact of media censorship and manipulation of epidemic-related facts in China is yet to be seen. Still, the short-term international unrest and dissent have only deteriorated international relations. At the risk of quoting a cliché as well as the established Streisand effect, the truth always comes out, no matter how many countries try to hide it.
The Spanish Flu spread was aggravated due to the failure of policymakers in adopting effective containment measures. Research shows that US cities which undertook measures to reduce contact amongst citizens in early-1918 displayed significantly lower peak death rates compared to cities that failed to or were too late to adopt disease containment policies.
To quote the results from this 2007 study, “Consistent with this hypothesis, cities in which multiple interventions were implemented at an early phase of the epidemic had peak death rates ≈50% lower than those that did not and had less-steep epidemic curves. They also displayed lower cumulative mortality.”
1: Contrasting the Death Rates in Philadelphia v. St. Louis during the 1918 Influenza Outbreak
As one might guess, Philadelphia was late to levy restrictions on gatherings, parades, and social distancing measures, whereas St. Louis was not.
From the same study, we see the definition of a popular phrase “flattening the curve” emerging – which is spreading out the rate of infection over time, enabling health care systems to treat people in a staggered manner, in line with available resources. And, to further corroborate this result, the following graph from the same study shows the different trajectories followed by the disease in two cities of the US.
Hence, regardless of the multiple debunking claims of social-distancing & lockdown as an effective preventive mechanism, previous epidemics have taught us that they perform a marvelous job at ‘flattening the curve’.
Social Distancing & Lockdown measures are effective, only if sustained, at least for a few months, even after things start to improve.
Markel (2007), shows the disastrous consequences that the early relaxation of bans on public gatherings in St. Louis had on death rates, during the Spanish flu. The pullback of social distancing measures to pursue economic interests was premature and caused a sudden, unprecedented surge in the number of deaths due to the flu.
2: Deaths due to the Spanish Flu in St. Louis – A Wider Picture. The black & grey lines show the duration for which social-distancing measures were active.
Another peak in death rates was observed only in cities that had relaxed distancing measures prematurely. Hence, it is crucial to consider this past artifact when governments compare the inconveniences to the public & damage caused to the economy due to such measures, with the benefits to the nation in terms of human life saved, and then make the call as to when the restrictions must be eased.
What can the industry learn?
The industry has played a key role in the obscure fight against the worldwide outbreak. But it faces its fair share of challenges, in such unprecedented times.
Global industries have weathered through multiple natural disasters and two epidemics of a similar scale and nature. Their recovery-paths have immense strategic implications for current policy-makers and offer strategic insights, helping companies chart out their short-term & long-term strategies.
3: This graph looks at the retail sales impact of the SARS epidemic in China in 2003; the earthquake, tsunami and Fukushima nuclear disaster in Japan in 2011; and the MERS epidemic in South Korea in 201.5
Three phases characterise the impact on the retail market in all these three cases:
The paths inevitably converge to stabilization but follow different trajectories, and are swayed by public perceptions and externalities, as in the case of the mid-autumn festival in South Korea. Looking at the retail sector from a micro-lens, past crises show that demand trends vary amongst product categories, both during and post the crisis. The three categories usually follow such trends:
Demand Trends during & post a Pandemic-generated Crisis
4: Demand Trends during & post a Pandemic-generated Crisis[1]
These three trends are a valuable tool for both large scale players and regional retailers who now have the means to prioritize and predict consumer-demand and consequently alleviate a sudden strain on the product supply chain arising out of sudden & unexpected demands.
A peculiar outcome of such epidemics have been the somewhat permanent changes instilled in consumer buying trends across the inflicted regions, across product categories – Fresh staples see a sharp rise in their demand and product hygiene & safety standards are ranked higher up in the consumers’ priority list. Hence, the advantage lies with producers who can quickly integrate changed consumer-preferences in their products, showcase product reliability & quality, and priced competitively, given the prevalent context. Lastly, previous pandemics have consolidated a long-standing theory, which is true not just for the retail sector but for any consumer-centric industry that there is, – Consumer Loyalty is tried and tested during these times. Effective Consumer Relationship Management (CRM) can forge bonds that go a long way, weathering through the thick and thin in such unprecedented times.
Implications
Almost two decades ago, the SARS outbreak in China had pushed entrepreneurs and established players to embrace the dawn of the e-commerce era. This outbreak is expected to accelerate further the shift from conventional brick-mortar based commerce to a centralized online cloud-store based mechanism. Survival, in the FMCG sector at least, will be decided to a great extent by the breadth of the supply chain networks and the ability to overcome severe bottlenecks.
The past has shown that even though established large-scale players with deep pockets and extensive networks, have a likelier chance to tough it through these hard times, even the regional players & SMEs make it through, strengthened by community-support, empathy and most importantly, local relationships.
What did we fail to learn?
This global pandemic outbreak has made one thing painfully clear – there is a lack of global collaborative research on ways to combat the spread of such infectious diseases. In the words of Johan Neyts, professor of virology and president of the Belgian-based, International Society for Antiviral Research (ISAR), health authorities failed to incorporate the learnings of the SARS outbreak of the early 2000s, partly because of the economic crisis of 2008 which squeezed funds from any potential research that could have been undertaken and partly because of the lack of seriousness in authorities in considering the efforts involved in dealing with the future expected breaks of coronavirus due to its seven different strains in existence. 
The Way forward?
All of these lessons come with a caveat – Things change with time. The modes of communication have changed. Both information & travel is much more accessible now. We have fast planes & faster internet. We have better healthcare capabilities and a much better, if not perfect, healthcare system. In 1918, people didn’t even know that a virus was causing the pandemic until much after its eradication. Today, we already have extensive research going on for 70 different probable vaccines for the virus, with a record setting pace in clinical trials and approvals.
We are much better equipped than what we were 100 years ago, and at the same time, somewhat under-equipped to deal with it. But these lessons from the various similar crises that humanity has faced and risen from, act as a glimmer of hope in these dire times, and if put to good use, can aid in the fight against COVID-19. In the words of Howard Markel, “There’s never been a better time in human history to have a pandemic than today, with the exception of next week or a month later. You want to kick that can down the road, but it’s here today.”
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

