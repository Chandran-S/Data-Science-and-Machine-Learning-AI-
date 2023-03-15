
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/should-celebrities-be-allowed-to-join-politics/'



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
text = ('''Should celebrities be allowed to join politics?
In today’s day and age of hyper-connectivity, celebrities – dubbed by Fraser and Brown (2002) as people who are famous for being famous and may or may not always serve others sacrificially – live in glass houses that are open to the prying eyes of the public. Notification bars are ever abuzz with news about when these celebrities wake up, what they wear to the gym, what they eat, and where they go on their vacations. In the idealized and idolized eyes with which society views celebrities, this unceasing visibility gives them an all-pervasive influence. The dominance of celebrities in swaying opinions of their acolyte is well-established – from advertisements in the print and pixel mediums, brand endorsements, and integrations, the evidence of exploiting celebrity influence is vast. However, today more than ever before, this influence is creeping away from the world of consumerism and into governance, policy, and politics. With an increasing number of public figures overtly taking strong and stronger political stances, making moves charged with a political motive, and eventually entering the sphere of legislature and bureaucracy, the scope of having a public influence has come to mean much more than it used to, thereby begging the question – how well-placed are personage in the world of politics? Should celebrities be allowed to join politics?
The interplay between the seemingly contrasting worlds of celebrities and policy-makers have come into increasing contact, in India and the rest of the world-leading pop-music artists back presidential candidates, internationally acclaimed actors attempt international diplomacy, films are claimed to be vehicles of political messaging, Prime Ministerial banquets engage celluloid stars, and so on. The ‘influence’ that stars exert on the world of politics seems to encapsulate an entire spectrum of involvement – at the zenith of which are the celebrities who actually set foot into parliamentary assemblies and councils, to represent the public that they once only influenced.
Although the phenomenon of the seeping of public influence into public policy has stretched through much of the 20th century, Choi and Berger (2001) suggest that the celebrity-politics connect continues to strengthen, much more than it did a few decades ago, as we trudge further into the century. Of the myriad causalities that underlie this circumstance, the global internet, where fame isn’t restricted to the confines of tangible achievement has been theorized to be a significant driver. In her 2009 book, Hyde labels this spectacle as a ‘mission creep’, where the enterprise of celebrities is expanding beyond its primary goals.
A prodigious public following seems to be the parallel between the worlds of politics and personage, which paves the path for celebrities to cross over to politics. Celebrities seem to be able to transition into politics with a certain grace, having already learned to live in the public’s eye, and established themselves, allowing the carry-over of their already-known traits as celebrities into their role as policy-makers. The visibility that autochthonous politicians work years to nurture, celebrities already have as they descend into the political sphere. In one of the best-known instances of a celebrity transitioning into politics, Ronald Reagan walked away from Hollywood and into the White House, carrying with him his trademark charismatic on-screen personality that spilled over into his politics. Initially serving as the governor of California for 15 years in the late 1960s, Reagan served two terms at the Oval Office. The voters knew who Ronald was – those who closely followed the trends of the red and blue – but also those who did not, giving Reagan the edge that propelled him towards success. On the other side of the globe, India saw much the same trend as an on-screen persona aiding the creation of an off-screen political career. Ramanand Sagar’s television series ‘Ramayan’, based on the Hindu mythological epic, was revolutionary in the pious Indian society of the 1980s when television shows were still novel. It has been recounted on many an occasion that audiences would often offer their prayers to the television screen, every time Arun Govil, who portrayed the deity ‘Ram’ made an appearance on screen. Having garnered a devout, dutiful fan following, Govil, and his other co-actors from the show turned to the world of politics, where their godly association in the public’s eyes helped them make a mark. Even the Philippines has shown ample evidence of celebrities time and again swooping in to claim political victory with their stardom and popularity, that their opponents lack.
Very often, the sentiment of anti-incumbency seems to be the golden ticket that lets celebrity politicians snatch political power from consummate politicians. Indeed, various political theorists suggest that anti-incumbency happens to be amongst the more effective strategies to secure vote banks. Until 31st December 2018, Ukrainian actor, screenwriter, and director Volodymyr Zelensky only played the lead in a sitcom that tells the story of a schoolteacher who accidentally becomes the president. It was that winter that Zelensky claimed that he was contesting the presidential elections to “change the mood and timbre of the political establishment” and to “bring professional, decent people to power”. Zelensky took the premise of his wildly popular show from the reel to the real, as he crushed rival, a seasoned politician, and incumbent Poroshenko in a landslide victory. Has Zelensky proved a better representative of his people than his predecessor? Time is yet to deliver its verdict. But the gushes that he invoked in the people of Ukraine, and his plea to oust ‘the ruling class’ played their role in assuring Zelensky 73% of the plebiscite. In the early 2010s in India, similar anti-incumbent sentimentalities were played into by the many celebrities including musician Vishal Dadlani, hockey legend Dhanraj Pillay, singer Jaspinder Narula, and others – all of whom joined hands with Arvind Kejriwal’s  ‘Aam Aadmi’ movement that aimed to rid the existing political system of its ills.
And so, innumerable celebrities across countries and cultures claim the throne. In fact, a Wikipedia page that maintains an inventory of actor-politician spanning the atlas claims that it ‘may never be able to satisfy particular standards for completeness’ – depicting the sheer number of celebrities who reroute to politics. This begets questioning the congruence of the world of personage and that of politics. 
In a research article published in 1995, Meyer and Gamson articulated the possibility of celebrities choosing to politically address only broadly acceptable social causes because they understand their power rests with staying popular, thereby failing to make any appreciable difference. This seems to ring true with some celebrity politicians in India, especially ones who are nominated to the Upper House of the Parliament but fail abysmally to leave a mark. One of many such representatives is Gautam Gambhir, a cricketer-turned-politician who has mostly been shunned for his abject performance in the parliament. Hema Malini, another popular actress who turned to politics, came under heavy criticism when her claim to be a protector of women’s rights wasn’t backed by an effort to fulfill her promises. Owing to such instances, research findings such as those by Nisbett and DeWalt (2016) find that the public often questions the intentions of the celebrities who venture into politics. It may also seem that in entering politics and failing to deliver, celebrity politicians often deprive grass-root political workers of the chance to enter the mainstream of politics.
However, there is ample evidence from around the globe that goes to prove that celebrities can, in fact, make their mark in the world of governance, independent of their primary identities. Take the South Indian state of Tamil Nadu, for instance, where the political ethos has been consistently overshadowed by celebrities from the world of showbiz, for at least the last 40 years. As per data from the Ministry of Statistics & Programme Implementation, Tamil Nadu is amongst the top-performing states in the country, having the second-highest GDP. Most celebrities in Tamil Nadu who approached politics have exhibited extraordinary political will and caliber. The late actor-politician J. Jayalalithaa, or ‘Amma’, ran successful terms as chief minister and is still considered a mother figure by hordes of her followers. Another stalwart in the landscape of Tamil Nadu, Dr. MG Ramachandran, continues to be celebrated as one of the most influential chief ministers and is lauded for his large-scale reforms. In being the forerunner of some of the changes that propelled the state towards progress, including easing caste tensions and other communal violence, ‘MGR’ as he is better known, is remembered as much for his governance as he is for his innings on screen.
More often than not, celebrity engagement in politics has been used to woo voters from the younger strata of the population. Empirical studies such as the one by Austin, Vord, Pinkleton, and Epstein (2008), have conclusively shown to enhance political engagement by younger voters.  Along similar lines, Jackson (2008) found that young people are more likely to agree with political positions that are endorsed by politicians. In fact, many modern political theorists are of the belief that the involvement of celebrities in politics allows for otherwise uninvolved or apathetic veins of the demographic, to be active participants in democracy. Hillary Clinton aimed to do just that when she got the Billboard-topping artists Beyonce and Jay-Z to take center-stage at her rally in Cleveland, in the weeks leading up to the 2016 US presidential elections – in fact, as multiple sources suggest, this rally did see a surge in millennial supporters. In such a case, it might be in the benefit of the community to have more celebrities endorse and engage in politics in order to nurture a more vibrant and engaged youth.   
From our vantage point today, the relationship between personage and politics seems to be a central fixture in society and is here to stay. What remains to be done, however, is to ensure the symbiotic course of this relationship – to scrutinize and critique the celebrities who knock the doors of politics and to secure their promises of beneficence and non-maleficence to the public that it seeks to serve. Yet, it may seem that these are benchmarks that need to be met by any and all servants of the people – celebrity or not, but in the world that we live in, this is certainly not the case. Politicians – celebrity or not, largely happen upon the sphere of governance without a lot of experience backing them and learn to navigate the maze of politics on the job. Ultimately, like all things in democracy, the government is of the people and its representatives are of them, and for them. The eventual endorsement of its leaders – celebrity or not – can come, not from data-based analyses and elaborate critique, but from the people. Should celebrities be allowed to enter politics? The question remains unanswered. Or rather, is answered by the democratic exercise itself – with the beeps of electronic ballots election after election, when the people – the real ‘influencers’, choose.
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

