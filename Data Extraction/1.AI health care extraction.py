
import requests
from bs4 import BeautifulSoup

# Example URL
url = "https://insights.blackcoffer.com/ai-in-healthcare-to-improve-patient-outcomes/"

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
text = ('''AI in healthcare to Improve Patient Outcomes
Introduction
“If anything kills over 10 million people in the next few decades, it will be a highly infectious virus rather than a war. Not missiles but microbes.” Bill Gates’s remarks at a TED conference in 2014, right after the world had avoided the Ebola outbreak. When the new, unprecedented, invisible virus hit us, it met an overwhelmed and unprepared healthcare system and oblivious population. This public health emergency demonstrated our lack of scientific consideration and underlined the alarming need for robust innovations in our health and medical facilities. For the past few years, artificial intelligence has proven to be of tangible potential in the healthcare sectors, clinical practices, translational medical and biomedical research.
After the first case was detected in China on December 31st 2019, it was an AI program developed by BlueDot that alerted the world about the pandemic. It was quick to realise AI’s ability to analyse large chunks of data could help in detecting patterns and identifying and tracking the possible carriers of the virus.
Many tracing apps use AI to keep tabs on the people who have been infected and prevent the risk of cross-infection by using AI algorithms that can track patterns and extract some features to classify or categorise them.
So how does AI do that?
IBM Watson, a sophisticated AI that works on cloud computing and natural language processing, has prominently contributed to the healthcare sector on a global level. Being a conversational AI, since 2013, Watson has helped in recommending treatments to patients suffering from cancer to ensure that they get the best treatment at optimum costs.
Researchers at Google Inc. showed that an AI system can be trained on thousands of images to achieve physician-level sensitivity.
By identifying the molecular patterns associated with disease status and its subtypes, gene expression, and protein abundance levels, machine learning methods can detect fatal diseases like cancer at an early stage. Machine Learning (ML) techniques focus mainly on analyzing structured data, which can further help in clustering patients’ traits and infer the probability of disease outcomes. Since patient traits mainly include masses of data relating to age, gender, disease history, disease-specific data like diagnostic imaging and gene expressions, etc, ML can extract features from these data inputs by constructing data analytical algorithms.
ML algorithms are either supervised or unsupervised. Unsupervised learning helps in extracting features and clustering similar features together that further leads to early detection of diseases. Clustering and principal component analysis enable grouping or clustering of similar traits together that are further used to maximize or minimize the similarity between the patients within or between the clusters. Since patient traits are recorded in multiple dimensions, such as genes, principal component analysis(PCA) creates the apparatus to reduce these dimensions which humans could have not done alone.
Supervised learning considers the outcomes of the subjects together with the traits, and further correlates the inputs with the outputs to predict the probability of getting a particular clinical event, expected value of a disease level or expected survival time, or risk of Down’s syndrome.
Biomarker panels that are mostly used to detect ovarian cancer, have outperformed the conventional statistical methods due to machine learning. In addition to this, the use of EHRs and Bayesian networks, which are a part of supervised machine learning algorithms, can predict clinical outcomes and mortality respectively.
Unstructured data such as clinical notes and texts are converted into machine-readable structured data with the help of natural language processing(NLP). NLP works with two components: text processing and classification. Text processing helps in identifying a series of disease-relevant keywords in clinical notes and then through classification are further categorized into normal and abnormal cases. Chest screening through ML and NLP has helped find abnormalities in the lungs and provide treatment to covid patients. Healthcare organizations use NLP-based chatbots to increase interactions with patients, keeping their mental health and wellness in check.
Deep learning is a modern extension of the classical neural network techniques which helps explore more complex non-linear patterns in data, using algorithms like convolution neural network, recurrent neural network, deep belief network, and deep neural network which enables more accurate clinical prediction. When it comes to genome interpretation, deep neural networks surpass the conventional methods of logistics regression and support vector machines.
Sepsis Watch is an AI system trained in deep learning algorithms that holds the capability to analyze over 32 million data points to create a patient’s risk score and identify the early stages of sepsis.
Another method known as the Learning-based Optimization of the Under Sampling Pattern( LOUPE) is based on integrating full resolution MRI scans with the convolutional neural network algorithm, which helps in creating more accurate reconstructions.
Robotic surgery is widely considered in most delicate surgeries like gynaecology and prostate surgery. Even after striking the right balance between human decisions and AI precision, robotic surgery reduces surgeon efficiency as they have to be manually operated through a console. Thus, autonomous robotic surgery is on the rise with inventions such as robotic silicon fingers that mimic the sense of touch that surgeons need to identify organs, cut tissues, etc., or robotic catheters that can navigate whether it is touching blood, tissue, or valve.
Researchers at Children’s National Hospital, Washington have already developed an AI called Smart Tissue Autonomous Robot (STAR), which performs a colon anastomosis on its own with the help of an ML-powered suturing tool, that automatically detects the patient’s breathing pattern to apply suture at the correct point.
Cloud computing in healthcare has helped in retrieving and sharing medical records safely with a reduction in maintenance costs. Through this technology doctors and various healthcare workers have access to detailed patient data that helps in speeding up analysis ultimately leading to better care in the form of more accurate information, medications, and therapies.
How can It help in Biomedical research?
Since AI can analyze literature beyond readability, it can be used to concise biomedical research. With the help of ML algorithms and NLP, AI can accelerate screening and indexing of biomedical research, by ranking the literature of interest which allows researchers to formulate and test scientific hypotheses far more precisely and quickly. Taking it to the next level, AI systems like the computational modelling assistant (CMA) helps researchers to construct simulation models from the concepts they have in mind. Such innovations have majorly contributed to topics such as tumour suppressor mechanisms and protein-protein interaction information extraction.
AI as precision medicine
Since precision medicine focuses on healthcare interventions to individuals or groups of patients based on their profile, the various AI devices pave the way to practice it more efficiently. With the help of ML, complex algorithms like large datasets can be used to predict and create an optimal treatment strategy.
Deep learning and neural networks can be used to process data in healthcare apps and keep a close watch on the patient’s emotional state, food intake, or health monitoring. 
“Omics” refers to the collective technologies that help in exploring the roles, relationships of various branches ending with the suffix “omics” such as genomics, proteomics, etc. Omics-based tests based on machine learning algorithms help find correlations and predict treatment responses, ultimately creating personalized treatments for individual patients. 
How it helps in psychology and neuro patients
For psychologists studying creativity,  AI is promising new classes of experiments that are developing data structures and programs and exploring novel theories on a new horizon. Studies show that  AI can conduct therapy sessions, e-therapy sessions, and assessments autonomously, also assisting human practitioners before, during, or after sessions. The Detection and Computational Analysis of Psychological Signal project uses ML, computer vision, and NLP to analyze language, physical gestures, and social signals to identify cues for human distress. This ground-breaking technology assesses soldiers returning from combat and recognizes those who require further mental health support. In the future, it will combine data captured during face-to-face interviews with information on sleeping, eating, and online behaviours for a complete patient view.
Stroke identification
Stroke is another frequently occurring disease that affects more than 500 million people worldwide. Thrombus,  in the vessel cerebral infarction is the major (about 85%) cause of stroke occurrence. In recent years, AI techniques have been used in numerous stroke-related studies as early detection and timely treatment along with efficient outcome prediction can help solve the problem. With AI at our disposal, large amounts of data with rich information, more complications and real-life clinical questions can be addressed in this arena. Currently, two ML algorithms- genetic fuzzy finite state machine and PCA were implemented to build a model building solution. These include a human activity recognition stage and a stroke onset detection stage. An alert stroke message is activated as soon as a movement significantly different from the normal pattern is recorded. ML methods have been applied to neuroimaging data to assist disease evaluation and predicting stroke treatment for the diagnosis.
Patient Monitoring
Today, the market for AI-based patient monitoring is impressive and monetarily enticing. It is evolving with artificial sensors, smart technologies and explores everything from brain-computer interfaces to nanorobotics. Companies with their smart-watches have engaged people to perform remote monitoring even when they are not “patients”. An obvious place to start is with wearable and embedded sensors, glucose monitors, pulse monitors, oximeters, and ECG monitors. With patient monitoring becoming crucial, AI finds numerous applications in chronic conditions, intensive care units, operating rooms, emergency rooms, and cardiac wards where timeless clinical decision-making can be measured in seconds. More advances have started to gain traction like smart prosthetics and implants. These play an impeccable role in patient management post-surgery or rehabilitation. Demographics, laboratory results and vital signs can also be used to predict cardiac arrest, transfer into the intensive care unit, or even death. In addition, an interpretable machine-learning model can assist anesthesiologists in predicting hypoxaemia events during surgery. This suggests that with deep-learning algorithms, raw patient-monitoring data could be better used to avoid information overload and alert overload while enabling more accurate clinical prediction and timely decision-making.
 Conclusion
Considering the vast range of tasks that an AI can do, it is evident that it holds deep potential in improving patient outcomes to skyrocketing levels. Using sophisticated algorithms AI can bring a revolution in the healthcare sector. Even after facing challenges like whether the technology will be able to deliver the promises, ethical measures, training physicians to use it, standard regulations etc, the role of AI in transforming the clinical practices cannot be ignored. The biggest challenge is the integration of AI in daily practice. All of these can be overcome and within that period the technologies will mature making the system far more enhanced and effective.''')

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

