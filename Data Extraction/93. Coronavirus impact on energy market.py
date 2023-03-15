
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/coronavirus-impact-on-energy-markets/'


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
text = ('''Coronavirus impact on energy markets
As the coronavirus spreads around the world and countries implement containment measures, energy supply and demand is being disrupted. The impact that country-wide lockdowns and demand slowdowns will have on energy markets will be evolving every day.
Impact on energy markets are:
i)Slump in wind expansion may support German power prices:
The recent slump in German wind capacity is likely to continue this year as complicated regulations and weakness in the global economic outlook due to the coronavirus are likely to dampen investment in the renewable technology.
This has the potential to restrict future additions to wind output, impacting supply and, in turn, supporting power prices.
ii)Chinese industrial gas demand remains weak in March:
Chinese natural gas demand from the industrial sector in March was down by almost 25% year on year, according to the Chongqing Gas and Petroleum Exchange released over Easter, as the impact of the coronavirus continued to weigh on consumption.
iii)Reduced energy offtake likely to hit UK suppliers:
A growing number of British energy suppliers could be at risk of colllapsing through reduced energy offtake and cash-flow constraints amid the coronavirus pandemic, according to industry sources polled by ICIS.
iv)Breakdown of European power demand impact :
Power consumption continued to soften across key European demand centres through March and the first week of April, while less stringent measures and colder weather has sheltered other parts of Europe.
v)Brexit set to ensure EU 2020 renewable goals:
While the coronavirus outbreak is already having a significant impact on the European renewables sector, one unintended consequence of the pandemic is likely to be the achievement of EU and member state binding renewable energy targets for 2020. This is due to the consequences of a prolonged downturn in demand, which will not be normalised for by the EU’s official statistics office (Eurostat), meaning that renewables look set to make up a higher share of consumption in 2020 across the EU.
vi)Spanish prices stay bearish but premium to France to persist:
Spanish power prices across the curve are expected to maintain a bearish trajectory throughout the nationwide coronavirus lockdown period which was recently extended until 26 April, with further extensions possible.
Nevertheless, forecasts for below-average wind generation would support Spanish spot prices, allowing France to continue exporting power to Spain. This in turn should keep Spanish near-curve contracts at a premium to their French equivalents.
vii)Coronavirus hits truck-loaded LNG demand:
Market sources have indicated the coronavirus has reduced demand for truck-loaded LNG around Europe, as countries have implemented lockdown measures and low oil prices have made gas less competitive as a transport fuel.
Data from terminals in northwest Europe shows the impact has not been as significant as originally thought.
viii)Europe has found a coronavirus electricity demand floor:
Electricity consumption has leveled out in Germany, France and Italy, after falling for several weeks due to restrictions aimed at stemming the spread of the coronavirus.
But in Great Britain and Spain, demand has continued to drop week on week in recent days, according to an ICIS model that controls for the impact of temperature across Europe’s five largest consumers of electricity.
ix)European power and carbon markets affected by COVID-19 – an early impact assessment:
In order to assess the fundamental impact of COVID-19 on European power and carbon markets we modelled a scenario in which we accounted for the first data on dropping electricity demand as well as assumed industry production cuts and continued travel restrictions.
x)India defers on LNG cargoes as lockdown ramps up:
India buyers are deferring LNG cargoes as ports close operations with strict lockdowns in place. This is a concern for sellers in a long market with India one of the key buyers.
xi)EU to reach 2020 renewables goals on coronavirus, Brexit:
The combination of Brexit and lower demand associated with the coronavirus pandemic should enable the EU as a whole as well as many member states to reach their 2020 renewable targets, according to ICIS model run.
xii)European power markets yet to hit coronavirus demand floor:
European power demand will continue to drop as governments escalate measures to stem the spread of the coronavirus.
Electricity consumption in Italy and France fell 16% below expectations over the past seven days, according to an ICIS model that controls for the impact of temperatures.
Three other major power markets studied by ICIS – Germany, Spain and the UK – have seen smaller drops, although all are trending down.
xiii)Virus demand hit to wipe 9% off 2020 European power prices:
• Power demand drop likely
• ICIS model indicates 6% drop in demand during 2020
• Price fall of 9% across Europe
A scenario where power demand drops one-tenth through up to June as a result of measures taken to slow the spread of the coronavirus would see power prices across European markets fall by an average 9% in 2020.
xiv)Weather-driven NBP prompt upside possible in week 13:
• Temperatures across NW Europe to tumble in week 14
• British local distribution zone demand to increase
• EAX-NBP premium gaining
Incoming cooler weather across Britain for week 14 could encourage buying across NBP prompt and near curve contracts in the coming sessions. The drop in temperatures will likely trigger a spike in heating demand which has also increased due to large numbers of people working from home to curb the spread of the coronavirus.
xv)Southeast Europe braced for falling demand, bankruptcies:
Eastern European and Turkish energy companies are braced for sharp drops in demand and potential bankruptcies as countries introduce emergency measures in a bid to contain the coronavirus outbreak.
Although the spread of the virus is not as extensive in this region as in Italy or Spain, the impact of the emergency measures that have been taken is likely to be felt acutely because national economies are fragile.
xvi)Europe on course for sharp reduction to power demand:
Power sector demand across key European countries is set for a major slump over the coming weeks as countries scale-up their efforts to tackle the spread of the coronavirus.
Demand in Italy, which has been hit the hardest by the outbreak, has dropped 10% compared to its five-year average for March during the second week of its nationwide lockdown.
As measures to tackle the spread of the virus intensifies across other European countries, a similar drop in demand can be expected, although changes are unlikely to be uniform.
xvii)China’s city gas demand steady despite coronavirus, uncertain elsewhere:
The impact of the coronavirus in China has reduced but gas demand will still end March well down from 2019. City-gate gas demand has been firm but the outlook for other sectors remains a concern.
xix)European gas centres set for April demand destruction:
• EU-wide demand crunch could follow after lockdowns intensify
• Residential demand to see brief increase before falling on warmer temperatures
• LNG sellers could struggle to deliver into Europe
xx)Coronavirus effects to ripple in Mexico:
Mexico’s economy has perhaps never been more exposed to the global supply chains and commodity markets hit by the shocks of the coronavirus and the oil price war, but its president is losing time engaging in denial and political pandering. These will not help his party’s 2021 election prospects if the country’s economic performance falls further as its currency takes a beating amid capital flight.
xxi)China’s gas demand rebounds – satellite data:
Emissions of nitrogen dioxide are on the rise in China. This is an indication that economic activity is rising as businesses and industry ramp up activity with the coronavirus impact subsiding.
xxii)Spanish virus to impact LNG imports:
Spain is home to one third of Europe’s LNG import capacity, so a lockdown has the potential to further gum up the global LNG supply chain if, as seems likely, it leads to a drop in demand for gas and a reduced ability to support scheduled imports.
xxiii)European nuclear plant operators gear up to ensure power supply:
European nuclear power availability is expected to remain robust with strict safety measures already being implemented by the major plant operators amid the coronavirus outbreak.
xxiv)Bulgarian power traders call for nuclear maintenance change:
Bulgaria’s free energy market association ASEP has called for measures to tackle national power market disruption due to the coronavirus outbreak on 16 March.
xxv)Global market moves feed energy downside jitters:
Heightened concerns surrounding the economic fall-out from the coronavirus is set to further weigh on European energy markets, traders say.
xxvi)Coronavirus likely to lower trader appetite for risk:
Widespread remote working due to the coronavirus is set to hit European natural gas and power liquidity, according to traders.
A number of firms have already asked traders to work from home or are expected to do so.
While some trading activities around gas and power dispatching need to be done from a trading floor, most other tasks can be done from home.
However, traders indicated it is unlikely to be business as normal with participants increasingly risk averse away from an office environment.
xxvii)Italian gas demand plunges on coronavirus restrictions:
The coronavirus outbreak took a toll on Italian gas demand in week 11 as the government extended nationwide restrictions against the spread of the virus.
If restrictions continue, demand and gas prices are likely to fall further as warmer temperatures follow spring, pressuring consumption for heating. Power prices are also expected to plunge on the back of weaker gas. Gas is Italy’s main source for power generation and the marginal fuel, therefore a price maker for Italian power.
xxviii)Expect weakness in Chinese summer gas demand:
• Coronavirus impact on LPG to worsen
• China’s big three will have sought some LNG volume deferrals as another means by which to deal with the mounting oversupply
• Incremental demand for gas and LNG this summer will be unusually low
xxix)Coronavirus fails to pull down Italian power demand significantly:
The Italian wholesale electricity market has so far avoided major losses following the outbreak of the Coronavirus in the country, the strongest so far across Europe.
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

