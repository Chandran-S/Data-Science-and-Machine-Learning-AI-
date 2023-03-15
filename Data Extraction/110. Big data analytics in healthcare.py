import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/big-data-analytics-in-healthcare/'
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
text = ('''Big Data Analytics in Healthcare
Quality and affordable healthcare is a vision of all governments across the globe. The high cost of Medical service and lack of sufficient infrastructure is detrimental to this vision. In developed countries such as the U.S., reducing healthcare cost is an agenda point of all the politicians.  Even in India, affordable healthcare will be as important as “Roti, Kapda and Makan” in the near future. 
Technology can play a big role in reducing healthcare cost and making it more efficient. The vast quantity of data is available with Hospitals and insurance providers. Big data can play a big role in analyzing this data to produce actionable insight. Some of the benefits of using big data are as follows,
Fraud has a big
impact on everyone. In fact, the National Healthcare Anti-Fraud Association
estimates that fraud costs Americans at least $33 billion to $55 billion
annually – that’s approximately 3-5% of the nation’s health care spending of
$2.26 trillion. For example: Sometimes, providers or hospitals report
procedures or services that were never provided. 
Big data analytics can help an insurance company to discover these frauds. Big data can use a huge amount of historical data to identify expected procedure or service for reported illness (identified using diagnosis code). If the expected procedure code doesn’t match with a billed procedure code, insurance companies can tag claim as a potential fraud claim. Once a claim is tagged as a fraud claim, the manual examiner needs to do additional verification to ensure that the claim is submitted by fraud provider.
Sometimes, patients’ fake identity to get the benefit of insurance. Using big data, companies can match patients’ information with services he/she has taken. If services taken by the patient is very different from the past medical history of patients, analytics can tag claim as one of the susceptible fraud claims.
Traditional health care providers find it difficult to store a vast amount of data. More ever, traditional database stores only structured data. Big data can solve storage issue by handling huge volume and variety of data. Having historical data of the patient can reduce healthcare drastically. 
It is very tough to get a Physician’s appointment in developed countries. One of my friends, senior director in a multinational bank, got an appointment for hip replacement surgery with a waiting period of six months. He was not able to sit and as a result, he didn’t drive or come to the office for six months. The waiting period can be drastically reduced if the hospital is able to forecast the number of patients likely to miss appointments. Based on historical data, hospitals can predict the number of no shows and can accordingly schedule appointments so as not to miss any slot because of no shows.
If a health insurance company pays an amount to hospital/provider that is greater than the contracted amount for service, payment is called an overpayment. Similarly, if payment made is lesser than the contracted amount for service, payment is called an underpayment. Overpayment or underpayment is a big headache for both insurance companies and patients. According to AHA data, private insurance companies have consistently overpaid hospitals for inpatient care for more than three decades. These overpayments have varied significantly over the years, but have grown dramatically in the past few years. 
Overpayment or underpayment can happen because of multiple reasons such as system glitch, a coding error. Using cognitive analysis, big data can predict the payment amount. If the predicted payment amount is significantly different than the actual payment amount, the system can identify possible overpayments or underpayments. 
Throughout the world, healthcare is based on diagnosing the presence of disease and subsequently treating the disease if present. Early diagnosis of the illness is a major challenge. Patients do go for diagnosis only when the symptom is evident.  It becomes challenging for the physician to treat illness once illness grows beyond a certain level.
Predictive or preventive can help to identify if an individual will develop a disease. Different forms of health device such as iWatch can capture data and send to the hospital at regular interval of time. Using predictive analysis, the hospital can predict if a patient is likely to develop an illness. Early diagnosis of illness can help doctors to treat patients effectively and at a lower cost. 
Predictive treatment can reduce the number of days a patient needs to be admitted in the hospital. As inpatient treatment is costlier than outpatient treatment, doctors can discharge patients if he/she thinks that the condition of the patient is not serious. The hospital can monitor the patient remotely using medical sensors and can call the patient to the hospital when it is required. 
Big data plays a big role when a person suffers from multiple illnesses. As per Dr. Abhishek Rai, M.S. Ortho of K.E.M. Medical College Mumbai, “It becomes challenging to operate patients when he suffers from multiple illnesses. Many a number of times, physicians are not aware of all issues with patients as patients themselves are not aware of all illness”. As an example, it’s tough to operate on a person suffering from diabetes as blood doesn’t coat easily for a diabetic person.  If a physician doesn’t have historical data of a diabetic person, he/she may operate diabetic person wrongly. 
Another example where big data analytics can play a huge role in predicting disease is a genetic disease. If a person has breast cancer, her child is 15-20 percent likely to develop breast cancer. Using big data analytics, the system can identify patients likely to suffer from genetic disease and request patients to do timely screening. Genetic diseases can be screened in first, second- or third-degree relatives to reduce the morbidity and mortality.  
In India, it takes 2-3 days on an average to get an MRI scan. Delay is not because of lack of medical instruments but because of lack of qualified radiologists. This problem can be mitigated by integrating Medical imaging with big data. Integration of medical imaging with big data will ensure that radiologists get the report in a short span of time. Big data will help not only to reduce the turnaround time required to get reports but also to get better reports. Technology can convert image to graphs and charts making it easier for radiologists to read the report. 
As data works on the principle of abundance instead of scarcity. So, the more data we have, the better the outcomes can be obtained. Thus, Big Data technology is clearly the digital disruptor to revolutionize the healthcare sector by consolidating the data from disparate platforms to build a 360-degree view about the human body and help the healthcare provider to generate solutions to life-threatening problems and improved healthcare opportunities.
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

