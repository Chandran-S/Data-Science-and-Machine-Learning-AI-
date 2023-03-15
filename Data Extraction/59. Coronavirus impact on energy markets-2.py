
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/coronavirus-impact-on-energy-markets-2/'


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
The coronavirus, also known as COVID-19, is not only a global public health emergency but also a source of significant regional and increasingly global economic disruption. This impacts the energy and climate world in many ways. The economic downturn puts pressure on global oil prices leading the Organization of Petroleum Exporting Countries (OPEC) to consider further cuts to production. It hurts the demand for natural gas during a time of extremely low prices. It changes the economic, energy, and climate policymaking environment in China, one of the most consequential energy consumers and sources of greenhouse gas emissions. And it has temporarily disrupted supply chains throughout the energy industry, including renewable energy, at a time when supply chain connections with China were being revaluated due to ongoing tariff and trade disputes. How consequential or transformative any of these changes are for the energy sector or for climate efforts will depend upon the ultimate trajectory of the virus outbreak itself
— Sarah Ladislaw, Senior Vice President and Director, Energy Security and Climate Change Program
The energy sector has already felt the impacts of the coronavirus. The outbreak has contributed to a dampened demand for oil, resulting in plummeting oil prices and production declines. As we move forward, then, the energy sector expects to face two headwinds: managing the issues of the health emergency all sectors face, and simultaneously coping with a low oil-price scenario, lower demand and the need to shore up revenue and manage debt obligations.
                               IMPACT OF COVID-19 ON INDIAN ENERGY MARKETS
In India, distribution utilities have a lower tariff for domestic and agricultural consumers, sometimes even below the average cost of supply, as compared to that for commercial and industrial consumers. Table 1 provides the electricity tariff rates in Delhi for selected consumer categories to highlight these differences. Thus, for several distribution companies, the lower tariff-paying consumers are cross-subsidized by commercial and industrial consumers.

The COVID-19 lockdown has led to shut down of all but essential commercial activities across the country. Approximately 1.3 billion citizens are obliged to remain within the confines of their homes and, in many cases, only allowed to work from home. Consequently, the electricity demand from industrial and, commercial customers has reduced significantly while the residential demand is expected to have increased. According to the Power System Operation Corporation of India (POSOCO), The energy met on March 16th, 2020 – which can be considered as a business-as-usual scenario – was 3494 MU as compared to 3113 MU on March 23rd, 2020 a day of voluntary curfew. It further reduced to a range between 2600-2800 MU between March 25th to March 31st, 2020. This trend is illustrated in the below Figure 1.
Thus, firstly, a key risk from the COVID-19 pandemic for the already struggling distribution companies in India arises from the loss of revenues due to the reduction of demand from the commercial and industrial customers as well as the inability to cover the cross-subsidies provided to the lower-tariff paying consumer. Secondly, the utilities would also have to account for the expense to comply with any ‘must buy’ commitments that they have with generators with long-term power purchase agreements. The true and full extent of this risk would only be known once a quantitative analysis is conducted when this crisis situation is contained. Thirdly, at an operational level, distribution companies would have to account for deviation in demand and supply patterns at a temporal and locational level. Finally, during this period, critical infrastructure such as electricity networks would have to be run with minimum employees.
As seen in Figure 1, the trade on the wholesale power market comprises just 4.3 percent of the total electricity transactions. However, the transactions through the power exchanges have grown over the last decade. The Indian Energy Exchange (IEX) has seen a growth from 2616 MU in FY 2009 to 52,241 in FY 2019.

Until now, the trade in the wholesale market is in four market segments:
1) Day-Ahead Market
 2) Term Ahead Market
 3) Renewable Energy Certificates
4) Energy-saving certificates.
 Recently, the Central Electricity Regulatory Commission (CERC) finalized the regulations for implementing real-time markets. This half-hourly market will enable the intra-day trade of electricity, allowing adjustment of generation and consumption profile during the day. Before the COVID-19 pandemic, it was announced by CERC that the real-time market would be operational from April 1st, 2020. However, the starting date has now been delayed by two months to June 1st, 2020. According to media reports, due to the COVID-19 pandemic, some required trials could not be completed. This delay in the real-time market implementation is likely to have a serious, adverse impact on the Indian power market.
Another impact of the COVID-19 pandemic on the power markets is in terms of the market dynamic. It can be observed that there is a dip in the clearing volume and the market-clearing price, which coincides with the gradually increasing shutdown measures taken by the government as a response to COVID-19 (See Figure 2). Thus, the reduction in demand due to the lockdown is reflected in the volumes traded on the electricity market and the clearing price.
Another point of reference is the price and clearing volume in 2019. On March 22nd, 2020, the day of voluntary lockdown, the clearing volume was 97.05 GWh, and the clearing price was 2195.48 ₹ /MWh. In comparison, on the same day in 2019, the clearing volume was 107.98 GWh, and the clearing price was 2816.18 ₹ /MWh. From the start of the lockdown, from March 25nd to April 1st 2020, the average clearing volume was 104.27 GWh compared to 130.24 GWh in 2019 during the same period. Similarly, the average market clearing price was 2155.93 ₹ /MWh in 2020 as compared to 3371.025 ₹ /MWh in 2019 for the same period.
 VIEWS OF THE SHAREHOLDERS IN INDIA
The lockdown has resulted in a shutdown of the industrial and commercial establishments and the stoppage of passenger railway services. This has adversely impacted the all India electricity demand, given that these segments constitute about 40% of the all India electricity demand, a statement issued by the rating firm said.
Further, these segments account for an even greater percentage of the Discoms sales revenues given that they are the subsidizing segments. This apart, with the focus of state governments being on healthcare and relief measures, the likelihood of subsidy support to the Discoms getting deferred cannot be ruled out, it added.
ICRA Ratings Group Head and Senior Vice President – Corporate ratings Sabyasachi Majumdar said, “The lockdown imposed by the government is likely to adversely impact the all India electricity demand, with demand expected to decline by about 20-25% on a year-on-year basis during the period of lockdown. This would in turn adversely impact the revenues and cash collections for distribution utilities in the near term, especially given the consumption decline from the high tariff paying industrial and commercial consumers and likely delays in cash collections from other consumer segments. The revenue deficit for the Discoms is estimated to be about Rs. 130 billion per month, on an all India basis. This would in turn adversely impact the liquidity profile of the Discoms, increase their subsidy requirement, and lead to delays in payments to the power generation and transmission companies.”The power ministry on Friday issued directors to the Central Electricity Regulatory Commission to provide a moratorium of three months to Discoms on payments to power generation and transmission companies.
The power ministry on Friday issued directors to the Central Electricity Regulatory Commission to provide a moratorium of three months to Discoms on payments to power generation and transmission companies and requested state governments to issue similar directions to state electricity regulators.
The power generation companies are already suffering delays in payments by Discoms across the majority of states, with payment due of more than Rs 85000 crore as of November 2019 at all India level as per the data on PRAAPTI portal.
With COVID-19 lockdown accentuating the delays in payments, the availability of adequate liquidity buffer in the form of debt service reserve and undrawn working capital limits remains important from a credit perspective.
However, it said that relief measures such as a moratorium on debt servicing over a 3-month period as notified by Reserve Bank of India and expected moderation in the interest rate cycle would be a source of comfort in the near term. The timely approval of the moratorium by the boards of the banks and financial institutions remains crucial.
The revenues for power generation companies having long-term power purchase agreements (PPAs) with the state distribution utilities (Discoms) will be protected by the provision for capacity charges linked to plant availability in case of thermal and large hydropower projects and “must-run” status in case of nuclear and renewable power projects.Average monthly thermal PLF would further dip to 50-52% against 63% in the corresponding period of the previous year, due to a considerable drop in demand and consequently, power generation companies especially those without any long-term PPAs would be adversely impacted given the weakening of the power tariffs in the short-term / power exchange market, said ICRA Ratings Sector Head & Vice President GirishkumarKadam.The under-construction renewable power projects as well as EPC and manufacturing companies in the solar segment are likely to face execution delays because of disruption in the supply chain in India and labor availability, following the lockdown. Given the import dependency on China for sourcing PV modules, the execution timelines for the ongoing solar projects are likely to be affected by delays in the delivery of PV modules following the outbreak of COVID 19 in China.
This delay in turn would increase the pre-operative expenses and the overall project cost, which in turn would have an impact on the expected returns.
In this context, the MNRE has notified that time extension can be provided for all renewable energy projects, which are impacted by the supply chain disruption due to the COVID outbreak, under the force majeure clause.
“Given the execution headwinds amid COVID 19 affecting Q1 of FY2020-21 and assuming the normalcy thereafter, the capacity addition in the wind and solar segments together is likely to degrow by about 25%, thus estimated at about 8 GW against earlier estimates of 11 GW in FY2020-21,” said Kadam.
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

