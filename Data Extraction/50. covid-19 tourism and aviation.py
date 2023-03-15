
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/impact-of-covid-19-pandemic-on-tourism-aviation-industries/'



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
text = ('''Impact of COVID-19 pandemic on Tourism & Aviation industries
As the Coronavirus pandemic unfolds, most industries face problems they had never imagined or prepared for. The Aviation and Tourism industries face the highest stress as Coronavirus spreads because not only are they a (usually) a leisure spending but also the nature of the disease directly conflicts with the industries’ innate business models. In this article, we will look at the impact of COVID-19 pandemic on the Tourism and Aviation industries both from the demand side and the supply side.
The Aviation and Tourism industries has two major customers: tourists and business travellers. Let us base this essay in the India and see what the reaction of the customer segments is expected to be like.
Almost all tourists will avoid vacationing for the foreseeable future because of:
Now, let us look at different types of holiday revellers. Tourists can be foreigners or domestic tourists. According to Business Insider “In 2019, almost 10 million foreign tourists visited India — spending over ₹1,800 billion during January to November period.”. However, because of the pandemic, almost all this business has been lost.
Due to the reasons stated above most foreigners will not prefer to travel. Those who do dare to venture out, perhaps wooed by discounted deals to places which would have otherwise been out of their budgets, will possibly be warded off by the one of the following factors:
Moving our focus to domestic tourists, we subdivide our population according to income levels:
In order to attract well paying customers, the Aviation and Tourism industries will have to spend a considerable amount of money to become sanitation compliant which will push profit margins even lower. Most Aviation and Tourism activities made money not on individual margins but on collective volumes, thus, they are inherently incompatible with the concept of social distancing.
Because of COVID-19 pandemic, businesses have realized that more and more meetings can be handled over video calls, thus, the movement of business personnel will be restricted. There will still be some representatives who will fly out to conduct business, but the industry can expect a permanent drop in revenue from this sector.
Moving over to the supply side, the Aviation and Tourism industries face strong disruptions in their supply chain. Because the industry is a leisure expense, manufacturing of many supporting equipment like the adventure sports gears and even parts of planes themselves has stopped. Because of the drop in demand, it is going to be tough to get the supply chain back up again. Furthermore, workers are afraid to show up to their jobs because of the high risk of contracting the virus due to exposure to multiple customers from different geographies.
Let us now see some examples of how the industry has done through March and April.
The bookings vanished in mid-March. Property owners had bought or leased real estate to list on the app were severely affected. The sharing economy, like Uber, Lyft and DoorDash had taken a hit but Airbnb was worse off because their expenses include cleaning services, interior design (one-time spend), and property maintenance which are fixed costs.
Thus, because of the pandemic the revenue was gone but the costs exist. They have people who depend on the owner’s rental income which made the problem worse.
Hosts usually decide cancellation policy, but under extreme circumstances, like this one, the company decided to override all existing policies (many of whom weren’t strict) and gave full refund to the guests. The company got $2 billion loan and has helped out the owners, financially by paying 25% for cancelled bookings capping at 5k/host. They have helped some hosts by getting them eligible for small business loans and avoid eviction.
Thankfully, there are too few properties to cause a housing crisis, but breakdown could cause strain on lenders and undermine property values (all want to sell to avoid foreclosure or defaulting on loans). AirDNA states that the listing split-up for Airbnb is: 33% list single property, 33% owners list 2-24 properties, 33% hosts have 25+ properties.
Some state governments in America have banned short-term rentals. This hurt the Airbnb owners because people looking to quarantine outside their homes or near relatives could’ve generated revenues.
So, what are hosts doing now?
Many are discounting units and looking for long-term tenants (12 months). Many are planning to apply for a small-business loan, seek forbearance from banks, find long-term tenants independently of Airbnb and sell property.
According to the Wall Street Journal, roughly half of all US workers stand to earn more from the Coronavirus rather than their work pay cheques. However, it must be noted that some have not gotten their money due to bureaucratical issues.
This complicates reopening because workers don’t want to come back and expose themselves to the virus and earn less. But businesses want workers to come back so that their small business loans can be forgiven, and business can reopen. However, money in the consumer’s pockets means the economy expected to rebound quickly when businesses open. About 40 million Americans are now on unemployment benefits, majorly from restaurant, hospitality, and retail industries.
Congress chose a flat amount of relief because it was really time-consuming for payments to be calibrated to each worker’s lost wages. Their $600 federal payment corresponds to $15/hours wage, but 21 states follow $7.25/hours.
Most workers don’t want to sit at home and are anxious to get back to work, but right now staying at home is the smartest financial decision for their families.
Workers should be ineligible for unemployment benefits if a job is made available to them. But owners are reluctant to report workers to authorities and sever relationships with employees they may need more later in the year when tourism demand is expected to pick up. Without income support, low-wage workers would likely seek out other jobs, including side hustles and gig work, which could expose them and their households to the virus. Thus, according to the government, it was important to give such a large stimulus to the economy.
Studies of SARs suggested that people sitting close to infectious persons were at large risk. Combined with that, in long journeys, passengers may take off masks. Thus, the risks of transmission are large and airlines need to focus on figuring out how to prevent transmission in planes.
Currently, the airlines plan on using self-cleaning material, long-lasting disinfectant, touchless lavatories, UV light as a disinfectant, antimicrobial coating for frequently touched surfaces, and cleaning between flights.
However, even after these efforts, only 12% of people are flying (compared to last year, same time) as of 27th May 2020.
Aviation and Tourism industries have the curse of being a luxury expense and thus, as the Coronavirus pandemic spread, the entire industry jolted to a halt. They are one of the hardest-hit sectors and have to adapt to a new normal wherein the revenue levels are unlikely to match to pre-COVID levels and margins have also reduced. However, in with a bleak future, this industry is here to stay because people have to travel, and eventually they will vacation as well. Coronavirus has become a “survival of the fittest” test for this industry and the stakeholders will come out of this disaster with strong business processes.
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

