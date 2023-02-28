
import requests
from bs4 import BeautifulSoup

# Example URL
url = "https://insights.blackcoffer.com/how-machines-ai-automations-and-robo-human-are-effective-in-finance-and-banking/"

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
text = ('''How Machines, AI, Automations, and Robo-human are Effective in Finance and Banking?
We all hear day in and day out that we amidst a technological revolution. But do we know what this really means?
Before we understand how its going to impact us, let us first discuss what these terms really mean.
A technological revolution simply means that we are in a period where better and newer technologies replace the others to get the job done faster and better. We are in an era with rapid innovations where machines are being compared to humans.

So, then what is Machine Learning?
Machine Learning is basically the application of artificial intelligence into electronic systems to enable them to learn and enhance themselves without being programmed by humans. It is the evolution and development of computer programs that can access data and then use it to advance themselves. Whether you know it or not, you use machine learning-powered applications daily.
Now, what is Artificial intelligence?
At its simplest form, artificial intelligence is a field, which combines computer science and robust datasets, to enable problem-solving. 
In simple words, Artificial Intelligence is the technology that facilitates these machines to perform human like behaviour.
Just like every other industry, machine learning is playing its role in the finance and banking industry too. In most cases where a human would perform the same task by performing the same calculations or following the same process can be taught to the machine which can now perform it by itself.
Let us discuss a few examples of the applications that we might have come in our day to day running which are a result of machine learning in this industry:
Portfolio Management
In earlier days, an investor would need to consult a financial advisor to understand his/her risk appetite and advise accordingly. Today, using machine learning algorithms there exists the concept of a “Robo-Advisor” that requires any user to give certain inputs about their financial status and goals and calculates their risk tolerance and constructs and idle portfolio allocation for them. Young users today find this extremely useful rather than physically visiting an advisor and paying a fee for doing so.
Risk Underwriting
Underwriting is one of the core functions for most financial institutions especially banks and insurance companies where they are required to underwrite the risk of the customers before loaning out money or insurance policies. These underwriting activities are based on trends and thumb rules industrywide. The same has been introduced through machine learning which is able to underwrite risks today on a larger and more accurate scale.
Algorithmic Trading
Machine learning is a mathematical model that tracks market information, analyses massive data sources and study market conditions simultaneously to detect patterns which can be used for trading. This is humanly impossible to do in a fraction of time. Algorithmic systems can make millions of trades daily, often known as “high-frequency trading”. It is highly believed that deep learning is playing its role in calibrating real-time trading decisions.
Fraud Detection
With the increase in use and dependency on computers for financial transactions came the data security risk. There is an ample amount of valuable data stored online available to create potential risk. Machine learning thus helped in fraud detection by detecting anomalies in transactions and flagging them for scrutiny based on the risk factors defined by the institutions. Fraud identification in insurance claims, credit card payments, identity theft, account theft, are all areas in fraud detection that machine learning can help in.
Process Automation
Process automation is one of the most common applications of machine learning in finance. The technology has helped in replacing manual work, automate repetitive tasks to avoid redundancy, and as a result, increase productivity. Machine learning has benefitted these organizations to optimize costs, improve customer experiences, and scale up their services. Some examples of financial and banking firms using process automation are the use of chatbots, automated calls, paperwork automation, and gamified employee training.
Customer onboarding
In this highly competitive industry, customer acquisition and the customer onboarding process is highly relevant in building a good customer relationship. At any stage, during the onboarding, a slight inconvenience or delay can act as a barrier. Machine learning-enabled complete automation in this process for these financial and banking institutions. Today, from opening an account, filing for any application can be completed within a few minutes with utmost ease. With AI, customers’ behavioral patterns have been studied to improvise and make the whole process efficient and user-friendly.
Customer churn
With the multitude of offerings and availability of a plethora of options, customer stickiness is a big problem faced by financial firms. Customer churn forecasting is one of the best big data use cases. It helps in detecting customers who cancel their subscription and analyses the same to tailor products as per customer needs. Video streaming application, Netflix’s subscribers worldwide has continued to grow to reach 167 million through using machine learning analytics on their customer database.
Decision Making
Financial and banking institutions function on facilitating investments made by their customers. Organizations are constantly in search of customers from whom they can get more revenue. This is now possible through performing machine learning analysis on both structured and unstructured data which helps them make more informed decisions. It also analyses data from the website and mobile application to construct effective marketing campaigns for the targeted customers.
Future of Machine Learning in Finance
Financial monitoring, security analysis, prevention of money laundering, network security, investment predictions, personalization of customer service everything comes under the realm of the applications of machine learning in the financial and banking industry. Yet, this is just the tip of the iceberg, there is a lot more that is going to change in the future. It is now visibly imperative that while AI is beginning to create a wave of transformation across these industries and adapting to these changes s important for one’s survival. With smart technology applied everywhere, all financial firms are bound to turn into FinTech’s to stay relevant to the “silver tech generation” consisting of millennials and the GenZs.
Final thoughts
The financial services industry has entered the space of artificial intelligence and machine learning, and the pace is not surprising knowing the positive changes it has brought. Machine learning has the most use cases in finance than any other industry because of the available computer power and new machine learning tools. The greatest applications include simplifying customer engagement and accurate sales forecasting. It is only making this industry better and more efficient with each new adaptation. Machine learning algorithms have the capability to deal with a lot more than human capacity along with eliminating human error. As even the algorithms are constantly learning and innovating, they can serve as a bridge to a completely flawless automated financial system in the future. Nonetheless, the challenges of high cost and lack of resources that come along play a significant role in how early these firms can adopt these technologies. But even then, the future seems bright as the industry has enough adopters and prospects ready to explore.
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

