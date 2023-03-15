
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/marketing-drives-results-with-a-focus-on-problems/'



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
text = ('''Marketing Drives Results With A Focus On Problems
When
the British ruled India, many Indians
accepted to work under their policy, and they did not try to rebel against the British for
their presence in Indian lands; They recognized the benefits of British rule. What
was the factor that made Indians accept the
British government? The perception about
British-policies (about trading, pay-scale, and positioning in the military) had been
done through various marketing channels in such a way that it brainwashed people of the nation, and many were ready to accept the British. So the Marketing is like fire when smartly
handled; it can destroy your enemies, but when not it can burn you.  
Marketing is to design and develop a pathway from product or service to the customer through a product, price, place and promotion strategies. Marketing depends on a variety of parameters that can be classified broadly, boiling down to vital three. These are ‘What type of customer are marketing team is focusing on (based on gender, age, location, customer-class) (W1)’, ‘What are Product Specification or Applications you need to bring in front of customer (W2)’, ‘What is your Product Promotion flow path (W3)’, ‘How much time it takes to convey product information to customer (H1)’. These criteria are what I call ‘3W1H’ forms. These basics need to be taken care of while developing a marketing strategy for the upcoming product. However, are they enough? No. There is one more parameter called the Feedback mechanism (FM). Generally, machines use the feedback mechanism to check output generated is as per input is given or not and if it is not, the input is corrected to get the required output. When it comes to human beings, he/she is surrounded by multiple products daily through social media, TV advertising and so forth. Slight uncertainties on either side in observation can make a positive or negative impact on the product on the consumer’s mind. In the feedback mechanism, the marketing team needs to answer only one question “What kind of product image built in the customer mind through promotional activities?” If the product image not built as per expectation, then management has to review and work on 3W1H, as discussed earlier. So let’s elaborate how 3W1H approach works along with feedback mechanism. 
After deciding the customer range, you can work on other
activities. Since the customer is
conscious about purchasing the product and
when the product promoted, the first thing the customer will check is its
usefulness, and if his/her answer is ‘Yes,’
then the customer will start comparing with similar alternative products in the
market. Therefore, one should present the
product along with some unique thought
and express the product in such a way that its image imprinted into the consumer
mind. For example, the polio campaign in
India. The government had taken the initiative to make India polio-free by
spreading awareness through all types of social communication mediums, in which
Television played a vital role. Legendary
cinema actor ‘Amitabh Bachchan’ was selected as an ambassador for this campaign.
Initially, the advertisement published in which Amitabh pleaded people for vaccinating
the children within a fixed schedule. The advertisement circulated throughout the nation, and
no stone was kept unturned by the government
to make this campaign successful. However,
much to the surprise, advertising failed miserably to attract a deprived community in polio booth. This promotion failure was because of
non-consideration of the customer (W1)
and how advertise impacted its application on the customer (W2)? So after analysing failure, experts recommended changes
in their marketing campaign like, to focus on major customer (Here mainly
village woman who generally spend their time in home by watching TV), type of
impact on people (People who watched TV during that era, had an image of
Amitabh as ‘Angry young man’). So the campaign
improved their advertising by bringing out the same angry Amitabh instead of
his soft-image, which used in the earlier campaign. So in a very next polio
campaign number of people in the polio booth suddenly increased and when
feedback is taken from people about such a sudden change, they replied that we
don’t want to watch ‘Amitabh sir’ getting angry because of us.[i]
In
W3, organisation must focus on what kind of channel use to attract the customer. Currently,
there are a lot of platforms available by
which you can market your product. The main
agenda you should keep in mind is that how effectively you can engage the customer
through marketing, at very first, if you make an
impact on customer then only, he can show interest in your product. For
example, changes in launching trailers of
the upcoming films. Till 20th
century Television and newspaper were crucial
factors, but now along with television, marketing team take help of internet
resources like Youtube, Facebook,
introducing an article on the pre-trailer phenomenon. Increasing the range of promotional devices, broadcast formats and publishing platforms provide a broad range to promote the film and choosing right platform may lead to a massive hit in the box-office.    
For
the last factor (H1) time, the obvious question that comes to mind is that ‘Is time for advertising matters?’, While
watching the TV when an advertisement appears,
many times we change the channel instead of
watching the whole ad or ignore it altogether. You do have an option of ‘Skip ad’ when you watch anything online, so one must decide how much time is required
to deliver product information during its
promotion? Delivering product information
in less time, may create the possibility that the customer will watch it so it
can result in attracting more customers. So, the content will be the
king in the marketing. However, as one
must have guessed it, this will not work all the time. Other factors like the cost of the product,
application range, type of customer can also affect
the H1. For example, the customer would be
more conscious while watching higher-priced
product broadcast compare to the lower-priced one. So it would take more time
for the marketing team to convince the
customer to buy the expensive good than the lower-priced one.
After
implementing a marketing strategy as per 3W1H, we need to work on a feedback mechanism in which organization has to collect information about
product image generated in customer’s minds through dealers and using feedback-form. Not using the feedback mechanism may lead to failure in achieving
‘product sales target’ and can make the situation even worse. Take the example of ‘THE TATA Nano-MAKING OF WORLD’S
CHEAPEST CAR,’ which was mind-blowing one-of-a-kind projects in Indian
automobile industry. From the first phase of a marketing
campaign, TATA Nano was on so much
continuous limelight that this project was pleased by former American president
Barack Obama.[ii]
The curiosity about the product
was so high that the sale of then-leading
car model Maruti-800 was dropped by 20% immediately after unveiling the Nano.[iii]
However, while marketing the product, an organization built “Affordable, Garibo Ki Car” image in customer’s mind,
which made customers conscious about what will be their status-quo in society
after buying Nano? Thus, they started giving preference to two-wheelers over Nano. During the same time, some instances of Nanos catching fire came
into focus, which worsened the situation.
This case even exacerbates when the team
did not work on the feedback mechanism
and hence, was not able to change customer
perception about Nano. Later, when Nano fire problem was technically sorted out
but even then, the marketing team did not
make any impactful effort to market the improved safety of the car. Therefore, customer’s negative perception
remained stuck and the organization did
not even come close to achieving selling
figures of Nano. If the organization had used
feedback mechanism, it would have been an entirely different story. 
So
3W1H can be an essential factor while
developing a marketing strategy for the
product along with a reversible FM
method. This method can help us to
monitor Product marketing throughout the phases and can help the marketing team to
change strategy whenever it goes into the wrong
path.
[i]
https://www.thehindu.com/news/cities/Delhi/when-amitabhs-voice-did-the-trick-to-make-india-polio-free/article6257123.ece
[ii]  https://www.businesstoday.in/obamas-india-visit/nano-catches-worlds-most-powerful-mans-eyes/story/10144.html
[iii]  https://www.businesstoday.in/obamas-india-visit/nano-catches-worlds-most-powerful-mans-eyes/story/10144.html
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

