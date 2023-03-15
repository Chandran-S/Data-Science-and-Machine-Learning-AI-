import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/ensuring-growth-through-insurance-technology/'



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
text = ('''Challenges and Opportunities of Big Data in Healthcare
To begin with I shall first like to explain what big data is and why it has become so important in our lives. Big data is simply data, but with a huge size. Data that is not just voluminous but also growing exponentially with time. Such data sets are so large and complex that it is not possible to store or process them using traditional data management tools. So, how huge can this data be? Social media sites like Facebook generate more than 500 terabytes of new data every day in the form of photos, video uploads, text messages, etc. A single Jet engine can generate more than 10 terabytes of data in 30 minutes of flight time. With many thousand flights per day, generation of data reaches up to many petabytes. Data contained in these sets are not always structured. It can be semi-structured or even unstructured. When such is the size and dimension of data we can well imagine how complicated it must be to process and analyze it. Therefore, modern methods of data analysis are being developed and used to process the information contained in big data sets.
Now coming to the opportunities that Big Data provides, they are not just limited to healthcare. Big data analysis is now a disruptive technology which has intervened into numerous fields and proved its worth. Healthcare is no exception. Big data has the potential to change the entire dynamics of the healthcare industry and improve the quality of life of people. Healthcare industry is a very large and complicated system. It involves a lot of risks and always demands better care. However, when a large number of patients seek emergency care the complications as well as the cost rise exponentially. In India, many times, we even lack sufficient infrastructure to support the patients, there is a deficit of beds as well as doctors to provide treatment to the patients. For instance, when epidemics break out and lives are lost at a very alarming rate, we can easily gauge our helplessness. However, the scenario is certainly improving now. With the advent of digitization into the healthcare system, healthcare providers or practitioners are now having access to a huge amount of patient health data. This healthcare big data can be processed and analyzed to identify patient patterns more quickly and effectively. The information obtained can be extremely useful to figure out chronic health issues and provide preventive treatment plans well beforehand so as to curb that disease or disorder from occurring. This method is also known as predictive analysis and it is one of the most crucial benefits of big data in healthcare. This technology can help the healthcare industry in more ways than we can infer. Healthcare organizations that have implemented predictive analysis have witnessed a reduction in ER visits by providing support and care to patients and decreasing emergency situations.
Apart from reduced ER visits and timely treatment, there are many other benefits of big data and predictive analysis. Patients with high-risk life-threatening issues can be provided with more customized treatment facilities. Due to lack of data, there exists a lack of proper planning in hospitals. Under or over-booking of staff, lack of medical equipment, medicines, and other facilities are a result of inefficient budgeting and ill-management of finances. Using predictive analysis these problems can be solved which will lead to reduced costs and efficient management of finances. Better staff allocation and admission rate prediction shall facilitate improvement in daily operations of healthcare organizations. By using big data the effect of recency bias could be reduced as well. When we give more importance to recent events and tend to ignore the effect of the older ones, it may lead to incorrect decisions. This is known as recency bias. Big data also helps in preventing fraudulent activities which in turn prevents losses of insurance companies. 
While all of this is changing the healthcare industry for the better, it is not that easy to reap the benefits of big data. There are a whole lot of challenges and vulnerabilities attached to its implementation. One of the biggest challenges is security. Healthcare big data contains the personal information and health history of patients. Acts of hacking, cyber theft and phishing pose a serious threat to these databases. Such data could be stolen and sold for huge sums of money. Protection of the patients’ privacy hence is a serious challenge to big data implementation. Also, the data would contain external data apart from medical information. The organization, therefore, has to take care of privacy, legal compliances, and government policies. Privacy and security of patients have to be given utmost importance and no breach of any kind can be permitted.
The next challenge is data classification and modeling. The size of the data is massive and it is less structured and heterogeneous. Classifying such massive data to identify relevant information is a big challenge. Modeling of such unstructured data is equally difficult. 
Storage and retrieval is another major challenge. Huge cloud servers with sufficient space are required to store such voluminous data. Also, the speed should be high so that uploading of data can be done hassle-free. The way storage is a challenge, retrieval also is a matter of concern. Integrating the data and getting all relevant systems to link each other is a tough job. 
The next challenge is a major one in my opinion. Finding the right
talent who own the expertise to implement this modern technology is an arduous
task. Shortage of required talent is a crisis in the market today. Even after
hiring the right talent it is a challenge to retain them. Scarce resources like
data scientists are hard to find and even harder to retain. They are easily
poached by competitors. A good and efficient compensation strategy, conducive
work environment, high incentives, opportunities for career growth and development
can be some of the ways of retaining such intellectual talent. 
In a nutshell, we can conclude that while big data is a disruptive technology which will bring about landmark changes in healthcare dynamics, the challenges and vulnerabilities need to be addressed with the utmost care and sense of responsibility.
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

