
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/online-gaming-adolescent-online-gaming-effects-demotivated-depression-musculoskeletal-and-psychosomatic-symptoms/'


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
text = ('''Online gaming: Adolescent online gaming effects demotivated, depression, musculoskeletal, and psychosomatic symptoms.
A video game is any software program that can be played on a computing device, such as a personal computer, gaming console, or mobile device. Video games have been in existence since the early 1970s and have become increasingly popular, spanning different mobile (smartphones, tablets) and stationary (computer or console) platforms. Advances, particularly in mobile devices, have given birth to social networks and group gaming
Although the nature of gaming does not require physical stamina and therefore is not limited by factors such as age, gender, or fitness, it is most popular with adolescents. In 2018, the World Health Organization (WHO) classified gaming disorder in their International Classification of Diseases (ICD-11). The ICD-11 is a list of diseases and medical conditions that aids health professionals in making diagnoses and treatment plans for patients having various disorders. It should be noted that, the inclusion of gaming disorder in the ICD-11 by WHO has generated vigorous and sometimes contentious debate and discussion within the medical and mental health community. Although recognized as an area of clinical interest, the Diagnostic and Statistical Manual of Mental Disorders (DSM-5) published by the American Psychiatric Association suggests that more clinical research is required before Gaming Disorder is formally considered as a psychiatric disorder. Similarly, the WHO notes the inclusion of gaming disorder in ICD-11, to encourage more research into excessive gaming behavior, including its prevention and treatment.
What is Online gaming disorder?
2018 WHO draft 11th Revision of the ICD-11 denotes the disorder as a pattern of “digital-gaming” or “video-gaming” behavior characterized by impaired control over gaming activity, increasing priority given to gaming over other activities to the extent that gaming takes precedence over other interests and daily activities, and the continuation or escalation of gaming despite the occurrence of negative consequences.
Online Gaming disorder has the same similarities with Internet gaming disorder (IGD), which is a condition that the American Psychiatric Association (APA) noted in DSM-5 as an area in need of additional study. The APA does not currently recognize IGD as an official condition. For gaming disorder to be diagnosed, the WHO criteria require that the behavioral pattern of the gamer must be of sufficient severity that major and noticeable impairment and deterioration in personal, family, social, educational, occupational or other important areas of functioning is present for a minimum of 12 months.
According to the WHO definition, a person with online gaming disorder will demonstrate the following characteristics for at least 12 months; problems controlling control their gaming habits, seeing gaming as more important over other necessities and daily activities or work, continuing to engage in gaming even after its negative health and social problems has been identified or are evident. Further research shows that gaming disorders can also be linked to anxiety, depression, obesity, sleeping disorders, and stress. People who remain physically inactive for long periods because of gaming may also be at higher risk of obesity, sleep disorders, and other health-related issues, according to WHO [1].
Video game-related health problems can cause continuous strain injuries, skin disorders, and other health issues. Other problems according to Shoja et al. (2007), include a condition that could be termed video game-provoked seizures in patients with pre-existing epilepsy. The following health consequences of video gaming have been reported.
Video game playing is associated with eye problems. Extensive and fixed staring at a video game screen causes eyestrain because the cornea, pupil, and iris are not biologically equipped for chronic heavy viewing of digital images from electronic devices. The visual system strain from frequent video game use over extended periods may result in headaches, dizziness, and in some cases, nausea and vomiting. Interestingly, there is some research that shows that gamers have an enhancement of spatial distribution of attention, compared with non-gamers. This somewhat predictable practice effect occurs with both peripheral and central visual attention. For sufferers of amblyopia (dimness or blurring of the eyesight due to a fault in transmission from the eye to the brain) video games may be helpful.
Persistent gamers may also suffer from musculoskeletal problems. A survey of children indicated increased physical complaints associated with video game playing. Such complaints range from pain in the hands and wrists to back and neck. Furthermore, a case report involving a nine-year-old teenager referenced Playstation thumb. Playstation thumb is, characterized by numbness and a blister is caused by friction between the thumb and the controller from rapid and persistent gameplay. Using dermoscopy, dermatologists discovered hemorrhages and onycholysis (the loosening or separation of a fingernail) in a patient who presented with hyperkeratosis. Also, tendon injuries (tendinosis) of the hands and wrists from tendon overuse, is another health problem associated with video game playing. Furthermore, another case report in the New England Journal of Medicine reported a fracture of the base of the fifth metatarsal from people who play Wii video games. This condition has been termed (perhaps sarcastically), a Wii fracture. There are also postural problems associated with the persistent playing of video games, although, ergonomic measures (including chair to monitor position) could potentially improve postural problems associated with video game playing.
Playing video games consistently has been associated with obesity. This is maybe related to the lack of physical activity in players. Alternatively, it could be that those who are less physically fit because of obesity gravitate towards less physically demanding activities, such as gaming. In either case, several studies have linked television and video games and increased Body Mass Index (BMI). It has been estimated that children in the United States spend 25% of their waking hours watching either television or playing video games. Furthermore, children who watch the most hours of television or play video games have the highest incidence of obesity. If video games are replacing physical activities and young people do not participate in physical recreation or see playing of video games as a form of recreation, this may be part of the link between times spent while playing video games and a rise in BMI in teenagers. Evidence for this potential link gains some support from a German study reporting that boys who spend no more than 1.5 hours per day engaged in television watching or playing video games, were 75.4% less likely to be overweight than those who spend more than 1.5 hours engaged in the same activity.
In a 2011 study, an association between video game activity and an increase of (mostly junk) food intake was reported. Specifically, single video game sessions caused an increase in food intake, regardless of appetite. It has also been suggested that active video game play using two popular gaming platforms has the opposite effect. Other researchers found no evidence that more active video games would result in a beneficial outcome although the study did demonstrate an increase in the amount of physical activity within the children receiving the active video games.
Health concerns that video games may cause epileptic seizures started in the early 1980s. The first medically documented case of a video game-induced seizure was reported in 1981. In 1993, a story in the popular press (Sun newspaper) reported that a boy choked to death on his own vomit during a seizure triggered by playing a video game. Similar but less serious incidents were subsequently reported by news media around the world, ultimately motivating video game console manufacturers to include that epilepsy warnings in the instruction manuals for their gaming products. In 1994 it was reported that video games only cause seizures in people already predisposed to epilepsy and advised that people with a predisposition to epilepsy can greatly reduce the risk of a seizure by staying 10 feet or more away from the TV set and wearing sunglasses while playing video games. This is clearly an area in need of additional research.
The cost: benefit value of prevention versus treatment of addiction disorders has been known for many years. It is important and beneficial to make use of several types of strategies in combating gaming addictiveness or disorder. Effective strategies include: educating gamers about gaming behaviors and consequences on their mental health; treatment geared towards helping the gamer to control his/her urge for video games, recognizing and dealing with disturbing thoughts and learning how to cope without video games; intrapersonal and interpersonal counseling to help gamers to explore their identity, build self-esteem, and enhance their emotional intelligence outside the fictional world of gaming and learning communication and assertiveness skills needed in social interactions; family involvement, including counseling and discussion family and other relationships; and developing new lifestyles. The latter is particularly important and to paraphrase the well-known mantra of Alcoholics Anonymous, it’s all about, people, places, and things. To prevent the persistent playing of video games, people need to explore new activities, set new goals and metrics of achievements beyond video game scores.
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

