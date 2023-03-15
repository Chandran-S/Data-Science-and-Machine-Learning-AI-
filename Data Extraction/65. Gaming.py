
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/gaming-disorder-and-effects-of-gaming-on-health/'
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
text = ('''Gaming Disorder and Effects of Gaming on Health.
Perhaps the virtual illusion has become today’s “New Reality”.
Inquisitive about the conclusion? Let’s attempt to vindicate as the outline story unfolds itself.
Ever since the concept of gaming floated back in the 1970s a new dimension was put before human beings, intangible in nature, and paradoxically engulfed us since then. It laid an impact on multiple generations in different ways. The last decade witnessed a boom in this industry & with the advancement of technologies in Virtual and Augmented reality, the gaming world has become an integral part of our day to day experiences. With the intent of being regarded as a leisure activity, it has now percolated over various strata and has managed to gather the attention of teenagers and adolescents on a comprehensive scale.
As we explored the fabric of our Engineering batch mates, we came across this peculiar guy holding excellent academics with accolades of his names, seemed to be a nerd at a glance and street smart by nature. Concomitance happened to be the reason for experiencing the lifestyle of other personalities that included SAM as well.
SAM was not only fascinated by the games but used to spend tirelessly hours and hours in front of the white screen engrossed completely leaving the track of time. As our academics advanced, the intensity of involvement in gaming grew deeper and steadily it turned into addiction in no time. We realized the gravity as we were exposed to his change of behavior and shift of priorities due to his unprecedented absence during our lectures and practical. He was modestly reluctant to share about his whereabouts with his peers. The situation turned grim when he wasn’t able to cope up with the subjects and was socially disturbed while interacting. He was compelled towards the fantasy world and was trapped in the vicious cycle of a never-ending backlog of games. As time progressed his performance turned from bad to worse. He was struggling. It was clear as ice that he had something hidden behind that visage.
As sophomores, we were also involved in gaming but soon realized that it does more harm than good. A close cluster of friends decided to lend a helping hand to SAM and as they learned about him, the ground beneath their feet slipped away and froze their brain. He had completely lost the circadian rhythm of his sleep, ran into financial losses, aggression draped him, his relations were tarnished due to the habitual routine, and signs of depression were visible on the upfront.
Poker was the tartar that he had caught. The game of cards as most of us would perceive it, rather proved to be the fulcrum of SAM’s near future. In the quest of pleasure through online games, he entered the tunnel of gambling at a very tender age ignoring the risks involved in terms of financial verticals. He was suffering from Internet Gaming Disorder (IGD) & his condition deteriorated as days passed by. Apparently, from a healthy individual to a drained skinny one, it took a toll on his physical health. Mental disturbance sucked him and he grew insensitive towards his academics. The psychological negative consequences were due to the combination of online gaming and gambling platforms. During that phase, the peer group (gaming group) gained more influence than parental control. 
Days arrived when he ran short of money and went ahead to ask his friends in college for the funds. Since a large amount of money is involved in gambling, a situation exacerbated when SAM had his ethical values on the line and breached his boundaries to satisfy his hunger of gaming and leaped towards illegal means to fund his gambling.
The practical project was a crucial part of our Engineering curriculum, under which a team of four members was to be established and is expected to remain intact for a year. SAM was by no means in anyone’s good books. No one in the batch was ready to risk with SAM even though he possessed a brilliant acumen. He was abandoned. We were three friends already and were scouting for the fourth. With mutual consensus, we decided to go ahead with SAM. Initially, we worked together and collaboration seemed to be perfect. Unfortunately, he wasn’t able to clear his backlog exams and had to drop out. At present he plays Poker on a professional level, keeping aside his Engineering career and battling with his conscience that we may not be aware of.  
Earlier generations used to spend their time or rather invest their time into activities building up their passion, in some cases they pushed their cognition to such an extreme and contributed to a level that humanity as a whole is in-depth to date. Can we imagine our present without the existence of Sir Edison, Sir Newton, Sir Tesla, Sir Einstein, and many more that were engrossed in their domain and served the necessary impetus for our generation? No one knows that SAM could have been our generation scientist, hadn’t he chose that path.  
Not Everyone is SAM but perhaps we could miss our Einstein or Tesla! Maybe our Newton is waiting for the new upgraded version of the game instead of sitting under the tree. Our Mendeleev might be involved in arcade games! 
Feels so numb.  
Millions of teenagers are involved in online gaming platforms and the community is growing at a rapid rate every year. Presently the prevalence of IGD among the adolescent group was between 1.3% to 19.9% and males reported more prevalence than females. In the fiscal year 2019, there were 300 million online gamers in India. This number was estimated to go up to 440 million gamers by the fiscal year 2022. Overall, India ranked the highest in terms of the growth in online game downloads on app stores with a growth rate of 165 percent between 2016 to 2018. 75% of young people use a mobile phone to play different games, 21% use PC/Laptop. Games like PUBG that promote virtual violence calls for the stern attention of the responsible demographics. As per a study, online games are directly affecting the neurons, leading to chemical imbalance thus causing severe depression and anxiety attacks.  
As per the World Health Organization, depression is interrelated to physical and mental health and the word ‘depression’ that didn’t exist decades back has become prevalent among the youth, leading to more stress and dysfunction and worsening the affected person’s life situation.
Anything in excess is poison. So are games. 
In deed the virtual illusion has become today’s “New Reality”.
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

