import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/the-future-of-investing/'

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
text = ('''The future of Investing
An investment is a resource or thing procured with the objective of producing pay or appreciation. Economically, investment is to buy products that are not expended nowadays but are utilized within the future to make riches. Financially, an investment can be a money related resource acquired with the idea that the resource will give wage within the future or will afterward be sold at the next cost for a benefit. 
The desire of a return within the frame of wage or cost appreciation is the center preface of investment. The range of resources in which one can contribute and earn a return may be an exceptionally wide one. Hazard and return go hand-in-hand in contributing; moo chance, for the most part, implies moo anticipated returns, whereas higher returns are ordinarily went with by higher chance. At the low-risk conclusion of the range are essential ventures such as Certificates of Store; bonds or fixed-income rebellious are higher up on the hazard scale, whereas stocks or values are respected as less secure still, with commodities and subordinates for the most part considered to be among the least secure ventures. One can too contribute in something as ordinary as arrive or genuine bequest, whereas those with a taste for the obscure – and profound pockets – seem to contribute in fine craftsmanship and collectibles.
Investing in green technology – Worldwide fresh investment in renewable vitality expanded by 2% in 2017 with add up to exchanges moreover expanding 1%. The industry finished 2018 with modern ventures of $279.8 billion and exchanges totaling $393.8 billion. 2018 is on track to be another solid year for venture with add up to unused ventures within the industry at $211.4 billion through the third quarter.
Green innovation ventures are taking an assortment of shapes, with increments in wind control and electric vehicle improvements, the establishment of renewable control capacity coming to unused highs and noteworthy increments in open showcase speculations around the world. Over the globe in 2018, Asia-Pac is driving venture with solar-powered developments drawing within the most noteworthy financing. Thus, what was once unimportant see into the long-run has without a doubt presently gotten to be a reality as nations around the world are making considerable ventures year over year in green innovation.
Green innovation contributing, too alluded to as clean innovation contributing, regularly includes the choice of ventures in companies with maintainable and naturally neighborly hones and products/services. Whereas a few clean advances offer advancements that increment asset efficiency and proficiency, others diminish the natural effect. As green innovation proceeds to rise as a developing drive, a few solid industry clusters have developed with shifting levels of speculation as development patterns rise and alter.
Micro-investing – Investment isn’t saved for those with parts of additional cash or associations to money related proficient. Anybody can contribute, and you don’t need much to urge begun. In reality, pennies will do fair fine. How you inquire? A progressively prevalent way is through a developing drift, made a difference by the most recent in-app innovation, called miniaturized scale investing. As the title infers, small scale contributing permits you to contribute cash in little sums, regularly automatically. Most smaller-scale contributing happens through mobile-based stages. You’ll make investments—you truly do invest money, and more on that fair ahead—or check your account on the go along with your cellphone or tablet. The thought is that little speculations made at frequent intervals can include up without much exertion or wallet torment. We are a “right here, right presently” culture, and in case we are able to arrange a modern combine of shoes whereas strolling down the street.
Depending on the sort of app you employ, smaller-scale contributing can cruel distinctive things. One way to miniaturized scale contribute is to utilize an app that rounds up the dollar sums on buys you make on a credit or charge card and redirects the additional cash to a speculation account. (There are too miniaturized scale reserve funds apps that occupy additional alter to a reserve funds account. These accounts are for the most part FDIC safety net provider and, not at all like venture accounts, are not subject to showcase risk.) Using an app on your phone, you enter a few individual data, counting your credit or charge card number or multiple account numbers, since you’re frequently able to interface various cards to your account. You must reply a few extra questions planned to decide the sort of investments that fit your venture objectives and work together with your chance resistance. A few small scale contributing apps utilize calculations to make a hazard profile based on your reactions to different questions, at that point make suggestions on how and where you ought to invest your money.
At that point, you ended up as an investor. After you spend $7.50 for a sandwich at lunch, the app rounds up your buy to the closest dollar ($8) and naturally exchanges $0.50 to a venture account. Once your round-ups are large enough (say $5), you’ll be able to utilize that cash to buy stocks or other ventures, or the app might make buys for you naturally based on your venture profile.
Artificial Intelligence-Driven Investing – The chart underneath visualizes how an AI procedure can select stocks based on a better dimensional see of the world.
In this case, the target could be a portfolio of developing markets equities that shows cautious, esteem characteristics. The chart maps all recorded values within the rising advertise values (MSCI EM IMI) universe. Each security is plotted on the surface based on its facilitates to tall dimensional esteem and profit highlights, which are each characterized as a portion of the AI approach. 
The vertical pivot appears how protective these include combinations have been on normal between 2013 and 2017. The higher the crest, the more cautious in down markets. The more profound the trough, by differentiate, the less cautious. On the proper side of the visualization, a two-dimensional cut appears where well-known developing showcase names — portrayed by their esteem and profit characteristics — finished this period.
This approach characterizes a complex and advancing choice boundary, outlined by the maroon form, inside which stocks are chosen to develop a portfolio. This choice boundary speaks to a steady, defensive/value locale of the outline. It may be a nonlinear locale that’s tall dimensional and advances as showcase conditions alter. Stocks inside the choice boundary tend to display protective characteristics per se, but the strategy moreover recognizes stocks that tend to have future profit pay development potential as well.
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


print('percentage_complex_words', percentage_complex_words(text))  # Output: 60.0

from textstat import textstat

fog_index = textstat.gunning_fog(text)

print('FOG index:', fog_index)

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

sentences = sent_tokenize(text)
total_words = 0
for sentence in sentences:
    words = word_tokenize(sentence)
    total_words += len(words)

avg_words_per_sentence = total_words / len(sentences)
print('avg_words_per_sentence:', avg_words_per_sentence)

import re


def count_complex_words(text, complex_words):
    # Split the text into words
    words = re.findall(r'\b\w+\b', text)

    # Count the number of complex words
    complex_word_count = sum(1 for word in words if word.lower() in complex_words)

    return complex_word_count


complex_words = ['analyze', 'sophisticated', 'articulate']

complex_word_count = count_complex_words(text, complex_words)
print('complex_word_count:', complex_word_count)

words = word_tokenize(text)
word_count = len(words)
print('word_count:', word_count)

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

print('avg_syllables_per_word:', avg_syllables_per_word)

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

