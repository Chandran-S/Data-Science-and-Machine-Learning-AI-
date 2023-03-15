
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/will-we-ever-understand-the-nature-of-consciousness/'
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
text = ('''Will we ever understand the nature of consciousness?
Introduction
The definition of consciousness has been controversial for centuries, hence it is given the title of the ‘most familiar and yet mysterious aspects of our lives’. An idea of this concept would be an awareness in beings of their surroundings, themselves, and their own perception. The reason this part of our mind remains unascertained is that consciousness isn’t observable, unlike brain matter that is studied scientifically. the physical clarification of awareness is in a general sense incomplete: it doesn’t include what it feels to be the subject, for the subject. There likewise is by all accounts an unbridgeable illustrative gap between the physical world and our consciousness. As suggested by an incident of Eastern and Shamir traditions, consciousness is both universal and primal. Whilst we have made tremendous progress in understanding brain activity over the years, this research hasn’t been able to answer all the questions relating to the nature of emotions and experiences.
History
Beginning within the late nineteenth century, this was a time that once had psychological queries driven by a philosophical understanding of the mind, which was typically equated with consciousness. As a result, the analysis of brain and behavior naturally thought of the role of consciousness in behavioral management by the brain.
The Ancient Mayans were among the first to propose a sorted out feeling of each degree of consciousness, its purpose, and its worldly association with mankind. Since consciousness incorporates stimuli from nature as well as interior stimuli, the Mayans trusted it to be the most essential type of existence, equipped for evolution. The Incas, however, thought about consciousness as a movement of mindfulness as well as of worry for others too.
John Locke, an early philosopher, said that consciousness, and so individuality, are freelanced of all substances. He also detected that there is no reason to believe that consciousness is stuck to any specific body or mind, or that consciousness cannot be transferred from one body or mind to a different one. Karl Marx, another early thinker, denies the mind-body classification and holds that consciousness is jeopardized by the material eventualities of one’s settings. William James, an American psychologist, differentiated consciousness to a stream – unbroken and continuous despite several changes and shifts.
While the main center of a lot of the analysis moved to strictly note cable behaviors throughout the primary half of the twentieth century, analysis of human consciousness has grown staggeringly after the 1950s.
In Sigmund Freud’s psychoanalytic theory, we can see that he believed that all three levels of awareness- preconscious, conscious, and unconscious were responsible for one’s behavior and thinking. He believed that the mid itself was divided into three parts- the id, the ego, and the superego. The Id is present at birth, instinctual, and operates according to the pleasure principle. The ego underseals reality and logic and develops out of the id in infancy. Finally, the superego is an internalization of society’s moral standards and responsible for guilt. Now, the Id is regarded as unconscious, whereas the ego and superego are also conscious and preconscious. Freud constantly revised his own clinical qualities researches, however, and didn’t conduct scientific experiments and hence his work is heavily scrutinized, leaving the questions unanswered.  
Sigmund Freud’s theory differed from the other psychologists since his theories were more understandable and very easily conveyed to the people. Sigmund Freud’s work and hypotheses helped people shape their perspectives on youth, character, memory, sexuality, and therapy. However, his theories were subject to considerable criticism both now and during his own life. Whilst John Locke and William James took a more practical approach to the mystery by conducting experiments, Sigmund Freud didn’t provide any evidence to support his claims.
Brain
Today, the essential focal point of consciousness research is on understanding what consciousness implies both biologically and mentally. Issues of interest include phenomena such as perception, blindsight, brainwaves during sleep, and altered states of consciousness produced by psychoactive drugs.
A greater part of the test assesses consciousness by approaching human subjects for a verbal report of their encounters. However, to confirm the criticalness of these verbal reports, researchers must contrast them with the action that all the while happens in the brain —that is, they should search for the neural connections of consciousness.
Hope is to locate that noticeable action in a specific aspect of the brain, or a particular pattern of global brain activity, will be greatly predictive of consciousness mindfulness. A few brain imaging strategies, for example, EEG and MRI scans, have been utilized for physical proportions of brain activities in these examinations.
A few investigations have shown that movement in essential primary sensory areas of the brain isn’t adequate to create consciousness: it is workable for subjects to report an absence of awareness in any event, when areas, for example, the primary visual cortex show clear electrical reactions to a stimulus. Higher brain areas are viewed as all the more encouraging, particularly the prefrontal cortex, which is involved in a range of high order functions.
One mainstream theory implicates various examples of brain waves in creating various conditions of consciousness. Analysts can record mind waves, or drawings of electrical movement inside the cerebrum, using an electroencephalograph (EEG) and placing electrodes on the scalp. The four types of brain waves (alpha, beta, theta, and delta) each correspond with one mental state (relaxed, alert, lightly asleep, and deeply asleep)
Memory
Episodic memory can be regarded as the only form of conscious memory. This is because it is the capacity to consciously remember personally experienced events and situations in the past. The hippocampus located in the brain’s temporal lobe is responsible for this type of memory.
Consciousness also plays a part in important memory distinctions. One such distinction is the implicit and explicit characteristics; in which explicit memory is what you consciously know and implicit memory includes events you may not be conscious of.
Furthermore, several empirical findings suggest that declarative memory is related to consciousness as well; meaning that the retrieval and formation of this memory are connected to awareness. Working memory operates/maintains consciously perceived information as well since it temporarily stores and tampers with information whilst working on tasks.
The current ways of testing this information are lacking in several essential aspects, including spatial resolution, temporal resolution, or scope. Examples of such methods are PET, fMRI, EEG, implanted electrodes, etc.
PET and fMRI have temporal resolution problems, EEG is well-known to have localizability difficulties, and implanted electrodes whilst great in temporal and spatial resolution can only test a set number of neurons; that is, they are restricted in scope. Hence, huge numbers of our speculations, while testable on a fundamental level, appear to be difficult to test as of now.
Mental illness
Consciousness has an influence on the way we see objects around us, which encourages us to settle on choices about how to communicate with them. Experiencing difficulty perceiving objects is connected to a few problems, for example, agnosia (a failure to decipher visual data), Alzheimer’s disease, and autism. However, we actually don’t comprehend what visual data is basic for the mind to intentionally perceive an object.
Several different disorders of consciousness include locked-in syndrome, minimally conscious state, persistent vegetative state, chronic coma, and brain death.
Locked-in syndrome, otherwise called pseudo coma, is a condition wherein a patient is aware however can’t move or impart verbally because of complete loss of motion of essentially all voluntary muscles in the body aside from vertical eye developments and squinting. The individual is conscious and is able to speak with eye movements.
In a minimally conscious state, the patient has intermittent periods of awareness and wakefulness. Patients need to give restricted, however reproducible indications of consciousness of themself or their current circumstance. This could be following straightforward orders, comprehensible speech, or purposeful conduct.
In a persistent vegetative state, the patient has sleep-wake cycles, but lacks awareness, is not able to communicate, and only displays reflexive and non-purposeful behavior. The term refers to an organic body that is able to grow and develop devoid of intellectual activity or social intercourse
Like coma, chronic coma results generally from cortical or white-matter harm after neuronal or axonal injury, or from central brainstem sores. Usually, the metabolism in the grey matter decreases to 50-70% of the normal range. The patient lies with eyes shut and doesn’t know about self or environmental factors.
Brain death is the irreversible end of all brain activity, and function (including involuntary activity necessary to sustain life). The main cause is total necrosis of the cerebral neurons following loss of brain oxygenation. After brain death the patient lacks any sense of awareness; sleep-wake cycles or behavior, and typically look as if they are dead or are in a deep sleep state.
Future predictions and conclusion
As we can see from the history of scholar’s endeavors to study consciousness empirically, its nature is not one that can be defined using scientific methods. In the past, psychologists such as Descartes came up with dualistic theories that did not line up with the fundamental laws of physics. In the battle between realists and illusionists, taking sides is fundamentally impossible as the topic doesn’t allow for concrete evidence.
As we can observe from neural examinations of the brain to detect the causes of consciousness, there is no sensory sector of the brain that is the cause for one to be aware during a certain event even if there are clear signs of a reaction to a stimulus. Similarly, even though consciousness plays a major role in memory, we currently lack the facilities to research it fully. In the future, there is a possibility that we will be able to access such facilities, however, which could allow us to find the neural connections and its ties to mental illness as well. Not only would this help several people by making us aware of the causes of our perspective on certain things, but it would also give us a better chance of recognizing signs of possible mental illness or other issues beforehand.
As it seems, researchers continue to study this unknown part of our mind and once we are able to fund and recover from the global pandemic, we will be able to answer one of psychology’s most difficult questions.
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

