
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/estimating-the-impact-of-covid-19-on-the-world-of-work/'
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
text = ('''Estimating the impact of COVID-19 on the world of work
“Stay at Home”, “Crisis”, “Pandemic”, “COVID-19”
These are just a few terms that come to mind when you wake up every morning and listen to the world mourning. With the Coronavirus pandemic hitting almost 210 countries throughout the globe and having no possible cure for the foreseeable future, the state of the world is as dismal as it can ever get. The “Great Lockdown” as it is being called is having adverse effects on almost everything a common man does but more importantly having a grave impact on the world of work. In this article, let’s look at what is the current status of the working population throughout the world and what lies ahead in the future.
2020 marked the beginning of a new decade and a seemingly difficult one for various organizations across the world. The first quarter of 2020 brought with it, major issues for the workforce. The implementation of lockdowns in various countries such as India, Germany, Italy, China, the UK, etc. has made “Work from Home” a compulsion for everyone except those working for essential goods and services companies. The crisis has already led to a worldwide economic slowdown and a period of recession is looming on our heads. The International Monetary Fund (IMF) has predicted a shrink of 3% in the Global GDP indicating the extent of the crisis. The severity of the issue is being felt by various big companies in India and other major powers of the world. In an interview, Anand Mahindra, CEO of M&M Group said that the manufacturing and production sector has been hit the hardest with the lack of labor force. He mentioned the lack of wage workers who help in loading and unloading of material is making transport a major issue for his company, also hinting at the importance of these small workers on the macro-level. CEOs and Managers of various organizations are having a hard time dealing with this crisis. With almost all MNCs such as PwC, BCG, Deloitte, IBM, Infosys, etc. making “work from home” compulsory till around mid-May’20, the importance of technology is at its peak now. Whether it’s conducting meetings over Zoom or conference calls over Duo, staying at home and working is becoming normal as we move forward. Service-based companies are finding it easier to complete their tasks as compared to Product-based companies. According to the IMF the world is going to lose USD 9 Trillion in output in 2020 which is more than the combined output of both Germany and Japan.
The manufacturing sector is one of the worst-hit by the pandemic. China, the world’s largest exporter of goods and the epicenter of the COVID-19 pandemic has seen a significant fall in output in the first two months of 2020. Chinese Industrial Production has fallen by 13.5% and major manufacturers in China are facing problems relating to the shutdown and lack of workforce. Restrictions have affected the supply of Chinese companies like JCB, SAIC, Nissan to name a few. Organizations such as Samsung, Kia, Hyundai, and all other non-essential goods companies have been forced to shut down factories to comply with lockdown norms and for the safety of their employees. Indian manufacturing giants such as Bajaj and Tata have taken the toll due to the lockdown measures in India. Bajaj has been forced to cut employee salaries by 30% and worker salaries by 10% due to a lack of revenue. US-based smartphone and tech giant Apple Inc. has shut all its stores worldwide with its factories in China and the USA unable to produce new products for the time being. These are just a few examples and production has been hit globally in all countries and there is a lack of demand to compliment this, further enhancing the problem faced by producers.
The world of work also includes those who are willing and able to work but cannot find a job, i.e., the unemployed. The unemployment rate is at a staggering high across all nations affected by the pandemic. According to the US Bureau of Labor Statistics, the number of people filing for unemployment hit a record high close to 7 million people. China reported an unemployment rate of 6.2% in Jan’20 and figures are expected to get worse in many other countries. According to the International Labor Organization, more than 25 million workers may go jobless soon due to this crisis and lead to a significant decline in income. The financial crisis of 2008 caused a 22 million spike in unemployment worldwide and the nature of the current situation makes it more dangerous. The fall in revenues and effects on both supply as well as demand mean that companies are having a hard time maintaining their balance sheet and taking care of their liabilities. Lack of job opportunities from the next quarter and salary cuts for existing employees might be the way forward in the corporate world.
Amongst all the negative impacts of COVID-19 on organizations across the globe, there are a few opportunities that are ready to be explored after this crisis. Technology and Artificial Intelligence are key aspects that have been mentioned in the past to be the ones contributing to the growth and a rise in revenues for various businesses. In times of Social Distancing and a shutdown of physical services, the digital markets may prove to be game-changers in the near future. Offering products online and promoting through digital marketing will be at the forefront. The quarantine time must be utilized in order to improve existing products and increase the customer base of companies. Byju’s, India’s famous online learning platform is leading by example in this front by offering its application essentials and courses for free. This will help them expand their brand in the coming years and create a customer base. Cult Fit, a Bangalore based fitness startup has used technology to create an application helping fitness freaks to workout from home. ONEPLUS, the Chinese Smartphone brand launched its new flagship, the ONEPLUS 8 in an online event, and created a filter to let smartphone lovers do a virtual unboxing of the device as getting the device physically is not possible at the moment due to restrictions. Decathlon, America’s leading sports and fitness equipment brand started an online campaign, “We’re in this Together” and the hashtag “#PlayItSafe” to appeal to their customers and increase sales of their home workout equipment. These examples show how brands and organizations have realized the importance of this lockdown to expand their base and consolidate their position in the market as they battle out the crisis.
The COVID-19 pandemic has led to chaos and panic throughout the working world. But there are huge opportunities for companies to expand and improve in the future. All organizations need to have a designed plan in place to ensure smooth functioning once the lockdown is lifted. Social distancing and hygiene will become key for manufacturing industries. The Travel industry which is currently in its worst state will need to recover fast in order to ensure the safe movement of passengers once borders are reopened across the world. The Service sector will need to move to digital platforms and hype digital marketing. Artificial intelligence will play a key role with a reduction in workforce eminent due to lack of demand. Handling tasks with the help of AI and reducing the need for an employee will improve cost-effectiveness until revenues rise back to normal. E-Learning has evolved over the past few years and its importance has been realized over this quarantine period. All major colleges and schools have transformed into an online program in order to educate students. This change is likely to be carried forward with more and more Educational Institutes switching platforms. E-commerce has also been an integral part of the world and will become even more important in the coming days. As malls and markets are shut and are expected to remain shut, e-commerce websites will benefit from the same. It is also an opportunity for retailers and vendors to create their base online or sell through e-retailers in order to generate revenue. Small vendors and retailers will continue to be an important part of the supply of the essential good to households and improving sanitation in these places will be key. MNCs that have switched to the atmosphere of “Work from Home” will find it easier to connect large groups of people together at all times and ensure all employees contribute to the smooth functioning of the companies in the crucial days up ahead. Entrepreneurial minds will also be encouraged to put forth their idea and create a startup of their own to make the corporate world an improved space in the future.
The world is in a state of panic and it is important for all of us to be united in this fight against COVID-19. As employees continue to work from home and make study rooms their cabins, organizations need to adapt to this new style of working and realize its importance for the coming months until a solution is found for this problem. Internet and online services will play an important role in the future. This transition will lead to a better “Digital World” and improve accessibility throughout the globe.
STAY SAFE, STAY TOGETHER AND DEVELOP YOUR COMPANY FURTHER!
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

