
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/gender-diversity-and-equality-in-the-tech-industry/'

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
text = ('''Gender diversity and Equality in the tech industry
Gender diversity is an equal representation of all genders in workplaces. One of the most important sectors where there is under-representation of gender diversity is the technological industry. Gender diversity means you have an equal opportunity which is not limited by gender. But, the true reality is gender diversity is a far outcry in the world.
Gender diversity isn’t a new topic but an old and global phenomenon and technology industry is not a stranger to it.  Women are often underrepresented in the technology sector. To understand the gap, let’s look at some statistics.
After looking at these staggering numbers, we see how biased gender diversity is even in the technological sector.
With rapid industrialization and education about the importance of rights, gender diversity is very important in today’s world. Gender equality represents a society that has lesser violence and provides a safer world for everyone. It is also directly proportional to sustainable development and ensures human rights to everyone. As today’s markets are increasing along with customers’ preferences, to understand customer profiles and build products we need gender diversity in our organizations. Equal representation helps in a better decision-making process for a positive impact.
Men and Women invest back into their families and having gender equality and diversity helps build values into families and households. Empowering individuals not only help themselves but also the economy of the nations. Gender equality and diversity should be built into organizations’ beliefs and values.
Global Gender Gap Report (2017)
Perception: If we see the word “gender”, it is a socially constructed definition. And this definition changes as per different cultural norms. In a much broader term, our society is also not aware of the concept of binary and non-binary gender, which includes queer, Trans, and Intersex individuals as well. The socially acceptable thing is to have gender expressions as per our gender identity. So, when it comes to choosing their career, people tend to choose a job role that gives them higher social belongingness. E.g. We see more women(as compared to men) in nursing careers as it requires more feminine qualities.
There are different versions of this perception: which we can see either in the society(in terms of socially assigned gender roles) or in the workplace(in the form of gender discrimination, stereotypical thinking, sexism, etc.). E.g. As per their assigned roles in society, women are most likely to take care of their family and children, and this affects their income and career growth. The motherhood penalty is a term defined by sociologists which states about the inverse correlation between income level and the number of children, i.e. there is also an income difference between a mother and a non-mother employee. As per OECD data(2012), there is a 7% reduction in wages for women per child.
Lack of economic opportunities: Even in the tech industry, women are paid less than men. As per ILO data(2019), on average women are paid 20% less than men worldwide.
Even when it comes to promotion, men are preferred more. It is to be noted that the numbers are even worse when there is intersectionality involved. E.g., a transgender woman will be paid less than a white woman and so on. As per the National Centre for Transgender equality, one out of two transgender people faces adverse effects: including 23% were denied a promotion, 44% were passed over for a particular job position and 26% are fired from their workplace just because they are transgender.
We can improve this scenario from the perspective of different stakeholders who are involved.
For Society :
Creating Tech awareness from School level: As per the HDR report( 2017), no. of boys pursuing STEM program is 97% higher than girls. One way to improve this is to introduce more strong role models to advocate for this issue. Minority gender communities should be aware of the multiple job opportunities and career growth available in this field. These job roles are not even gender-specific anymore. For developing countries like India, STEM scholarship programs can be introduced from the secondary education level.
Address Bias/Stereotypes: It is to be noted that perceptions, stereotypes, and biases are not just something we learn in school but our upbringing also creates these, the things we watch, listen, and read daily. So, it also becomes the duty of the society to create a more inclusive environment. E.g. A more privileged person should fight for the fundamental right of the less empowered or less privileged person in society.
There should be active encouragement from parents, teachers, and educators for students in STEM programs regardless of their gender identity.
For Companies:
To promote gender diversity in the workplace, companies need to focus on three things:
Unbiased Recruitment policies:
Generally, there is a lot of unconscious bias and prejudices when hiring women, trans, intersex, and queer individuals. There needs to be a diversity and inclusivity team in the HR itself that addresses these issues. A blind recruitment drive can be conducted where there is no need to mention gender in any of the documents.
Having structured questions during the interview might help to remove any unconscious biases.
Diversity sensitivity training: Workplace policies need to create a company image that hires everyone irrespective of their gender identity and can even keep it optional in all the job formalities and documents.
Conducting proper sensitivity training to employees is essential as they need to use correct pronoun/gender-neutral language.
Not just sensitivity training, strong workplace policies need to be in place which deal with any form of harassment or gender discrimination.
To create accountability and transparency in their process, companies can also share diversity and inclusivity company data.
Retention policies:
Creating Mentorship models:
Including proper maternity leaves, child care facilities, and flexibility of working hours/remote working arrangements can be helpful for minority gender in keeping their work-life balance.
Develop support programs for these communities who are joining after maternity leave. It might be in terms of psychiatric or mentorship support.
Increase Pay parities: It can be done in the following ways:
Lastly, there needs to be proper government policies in place which provide a constitutional or legal framework starting from the basic education level. It may begin from increasing the gross enrollment rates of all students in schools (irrespective of their gender identity) for developing countries( like India) to have proper compensation policies in the workplace(as similar to what states like California, New York has adopted) to ensure pay parity.
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

