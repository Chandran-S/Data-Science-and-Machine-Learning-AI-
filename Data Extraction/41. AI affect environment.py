
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/how-does-artificial-intelligence-affect-the-environment/'

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
text = ('''How does artificial intelligence affect the environment
When we talk about AI, different people have different perspectives about it. There is one group of people, having good knowledge about the real potential of AI, who believe that AI can be a novel solution to many problems that the world is facing today. There is another group of people who are merely threatened by the thought of AI taking over the world.
The field of AI took birth when Alan Turing in 1950 had this thought “Can machines think?”. Later in the 1980s, with the adoption of “expert systems” by the companies around the world, the booming of the field of AI was initiated. Initially, it was a matter of awe for everyone to see the results of what AI can achieve. AI’s growth in the past few decades has been exponential and it has transformed the way we live, work and solve challenges. AI has made its impact in a wide variety of areas including healthcare, education, business and many more. But, one of the areas of highest impact that AI has made recently is in the environment and climate change.
Is AI Revolutionizing the way we deal with Environment and Climate Change?
AI can strengthen climate predictions, enable smarter decision-making for decarbonising industries from building to transport, and work out how to allocate renewable energy. In recent years, AI has been a game-changer in how many people, as well as organizations, deal with climate change. Microsoft believes that artificial intelligence, often encompassing machine learning and deep learning, is a “game-changer” for climate change and environmental issues. The company’s ‘AI for Earth’ program has committed $50 million over five years to create and test new applications for AI.
AI is increasingly used to manage the intermittency of renewable energy so that more can be incorporated into the grid; it can handle power fluctuations and improve energy storage as well. Wind companies are using AI to get each turbine’s propeller to produce more electricity per rotation by incorporating real-time weather and operational data.
AI can also improve energy efficiency on the city scale by incorporating data from smart meters and the Internet of Things (the internet of computing devices that are embedded in everyday objects, enabling them to send and receive data) to forecast energy demand. Besides, artificial intelligence systems can simulate potential zoning laws, building ordinances, and flood plains to help with urban planning and disaster preparedness. One vision for a sustainable city is to create an “urban dashboard” consisting of real-time data on energy and water use and availability, traffic and weather to make cities more energy-efficient and livable.
Hotter temperatures will have significant impacts on agriculture as well. Data from sensors in the field that monitor crop moisture, soil composition and temperature help AI improve production and know when crops need watering. Incorporating this information with that from drones, which are also used to monitor conditions, can help increasingly automatic AI systems know the best times to plant, spray and harvest crops, and when to head off diseases and other problems. This will result in increased efficiency, enhanced yields, and lower use of water, fertilizer and pesticides.
But, is AI as good as it looks or does it have a downside?
When we talk about the effect of AI in the environment and climate, both sides of the coin must be elucidated. So, apart from all the positives discussed above, the usage of AI also has a downside.
Last year’s World Economic Forum report showed that while AI can address some of Earth’s environmental challenges, it is important to manage it properly. According to the forum and experts in the field, AI has the potential to accelerate environmental degradation. The use of power-intensive GPUs to run machine learning training has already been cited as contributing to increased C02 emissions.
Although AI has been around for about half a century, the question of environmental impact – and other ethical issues – is only arising now because the techniques developed over decades can now be used in combination with an explosion in data and strong computational power.
For all the advances enabled by artificial intelligence, from speech recognition to self-driving cars, AI systems consume a lot of power and can generate high volumes of climate-changing carbon emissions.
A study last year found that training an off-the-shelf AI language-processing system produced 1,400 pounds of emissions – about the amount produced by flying one person roundtrip between New York and San Francisco. But there are ways to make machine learning cleaner and greener, a movement that has been called “Green AI.” Some algorithms are less power-hungry than others, for example, and many training sessions can be moved to remote locations that get most of their power from renewable sources.
The key, however, is for AI developers and companies to know how much their machine learning experiments are spewing and how much those volumes could be reduced.
What can be done?
To prevent this, the proposition by the World Economic Forum report is that advancements in “safe” AI should be pursued, to ensure that humanity is not developing AI that is harmful to the environment. Specifically, the World Economic Forum said in its report that AI developers “must incorporate the health of the natural environment as a fundamental dimension.” This means safeguarding against models that will demand the consumption of energy or natural resources beyond what is sustainable, among other factors. In a sense, all programs need to be designed with the dimension of environmental protection and improvement in mind.
In the future development of AI programs, it will also be important to note the environmental impact of creating these systems in the first place. According to an academic study on energy usage for deep learning processes, the creation of an effective AI might be costly to the environment. Nearly 300,000 kilograms of carbon dioxide equivalent emissions are created during the process of training a single model. This is basically equal to the emissions of five average cars in the United States. Considering the negative environmental impact in addition to the positive implications of AI and climate change will be crucial, moving forward.
There is always room for improvement, some pioneers of machine learning recently published a paper called Tackling Climate Change with Machine Learning was discussed at a major AI conference. David Rolnick, a postdoctoral fellow of the University of Pennsylvania said “call to arms” which means to bring researchers together.
This paper covers thirteen areas of research where machine learning can be used, including energy production, CO2 removal, education, solar geoengineering, finance and many more. The main idea is to build more energy-efficient buildings, creating new low-carbon materials, high tech monitoring of deforestation and greener transportation sources. Artificial intelligence and machine learning can be a medium but the real work is to be done by us as a caretaker of mother earth.
How about better forecasting of climate change?
 The kick start was already done by climate informatics in 2011 which regulates the collaboration of data science and climate science. It covers a wide range of topics like improving prediction of extreme events such as hurricanes, paleoclimatology, collecting data from ice cores, climate up-down scaling and its impact on us.
“There’s a lot of uncertainty,” Monteleoni
Claire Monteleoni, a computer science professor at the University of Colorado concludes that AI/ ML models generally forecast for the short-term. There is a significant difference when it comes to long-term forecasting.
The first system was developed at Princeton in the 1960s. All the AI/ ML models developed till now revolve around atmosphere, oceans, land, cryosphere or ice. AI can help to achieve new heights with the amount of data collected at this point of time, and compute more complex climate conditions using climate modeling algorithms.
What about showing the effects of extreme weather?
We all know that climate change is real and the rise in atmospheric temperature the proof, but to make it more realistic, Montreal Institute of Learning Algorithms (MILA), Microsoft and ConscientAI Labs uses Generative Adversarial Networks popularly know as GANs to simulate what mother earth will look like with the rising sea levels, atmospheric temperature, air pollution and intensive storms.
“Our goal is not to convince people climate change is real, it’s to get people who do believe it is real to do more about that,” said Victor Schmidt, a co-author of the paper and Ph.D. candidate at MILA
The project is deployed for the people to look at what their places will look like in the future when there will be a significant climate change.
Can we track carbon emissions?
Short answer, yes. Tracking of carbon emission is one of the agendas of the UN. For this year 2020, the UN’s goal is to prevent new coal plants from being built. Satellite images come into picture using deep learning techniques, data science researchers can analyze these high quality satellite images and track the emission. Google is expanding its horizon of nonprofit’s satellite imagery, including gas-powered plants emissions which can be used by researchers to track the pollution. There are countries organizations which govern CO2 emission on the ground level but these data are unreachable by global monitoring associations.
AI can automate the analysis of images of power plants to get regular updates on emissions. It also introduces new ways to measure a plant’s impact, by crunching numbers of nearby infrastructure and electricity use. That’s beneficial for gas-powered plants that don’t have the easy-to-measure plumes that coal-powered plants have.
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

