
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/how-to-increase-social-media-engagement-for-marketers/'


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
text = ('''How to increase social media engagement for marketers?
Social media is such a staple of our evolving digital culture that it’s almost hard to imagine a time without updates, likes, comments, and shares. People are more concerned about how their food looks than about how it tastes. Thus, it has become necessary for any brand to have a social media presence. The phrase “Out of sight, out of mind” has never been so apt than in today’s time. Social media is as much about engagement with other people as it is about sharing content. It’s why we call it “social” media.
Social media engagement simply is the interaction between the customer and the brand on social media platforms. Where else are you going to find a huge database of your target demographic who are already in the mind frame to read, click on, and share your message?
In practicality it is not about a one-off communication but more about the construction of a long-term relationship with your target customers. Hence, when we talk about such a relationship it is imperative to consider the positive actions of users as a reaction of any brand’s engagement tactics.
A fundamental customer journey can be envisaged in the form of customer funnel. The touchpoints available at every step can be shown as: 
To understand the reach of social media by the use of numbers
Most popular social networks worldwide as of July 2020, ranked by number of active users (in millions)
How many retweets on Twitter? How many reactions to the post on Facebook? How many likes on Instagram? How many shares on Linked In? This seems to be the new age Resume.
Thus, a good question or mystery for any brand is how to measure their social media engagement. As with most online actions, it’s often difficult to measure offline benefits. More often than not the online campaigns do not drive any conversions on the same day/week, but a user may see those posts and become more familiar with the brand. That familiarity, later on, might result in him choosing that brand over an unknown competitor. Social media engagement as a result helps businesses in developing their brand’s personality, improves visibility, and becomes its voice.
Customers exhibit dynamic behavior based on the type of product. It could range from picking and variety-seeking to high involvement processes comprising extensive information search. Top of the mind recall rules due to this purchasing behavior implies that targeted advertising and marketing is extremely important. No one engagement strategy can be applied across brands. Each brand’s strategy must differ based on what the brand is trying to sell, its target customers, and the beliefs of the brand.
Some of the key aspects of marketing strategy that we have influenced as vital cogs in the digital marketing wheel are:
Influencer Marketing
Influencer marketing involves a brand collaborating with an online influencer to market one of its products or services. Here the audience doesn’t care much about your brand but the opinions of the influencers.
An influencer as being someone who has:
2017 saw Chanel being named as the most influential luxury brand on social media. Since then, it has continued to grow and diversify its portfolio, while providing constant engagement to its plethora of followers, especially on Instagram. 
Cross-Channel Marketing
Cross-channel marketing or multi-channel marketing is the practice of using multiple channels to reach an audience. Because not everyone uses every single platform, it’s a good idea to have a presence on a few, and share some of the same content across the different networks. What makes a difference in this marketing strategy is the choice of platforms. Apart from using basics like Facebook, Instagram, and Twitter a clever use of Spotify, YouTube, or IGTV to showcase the brand can make a real difference.
The luxury vehicle brand, Land Rover’s cross-channel marketing includes the Google Display Network, homepage masthead and Masthead in Lightbox ads on YouTube, and visibility through mobile, search, and Google+. The brand’s digital campaign included four different influencers who created visual content for their blogs by taking trips to places like Glacier National Park and the Appalachian Mountains and Land Rover’s microsite. Their cross-channel efforts resulted in 100 million impressions from YouTube, and a 10% increase in search ad CTR. Further results found that online leads from its digital channel efforts now account for 15% of the brand’s total sales.
Informative
The content created on social media must add some value addition to the customers. The brand must constantly keep informing the customers about their existing products, new products, achievements of the brands, or even casual content. There should be “Wow, I didn’t know this” factor. Only then will they become regular followers of the brand.
An example of this would be a cosmetic brand posting make-up tutorials. Fenty Beauty is one such prime example, which has built up to 1.4m followers within just 4 days of its launch, eventually garnering close to 10m followers, with Rihanna being one of their icons.
Involving customers & engaging them:
“Customer is King” is a known concept. It is the foundation of traditional marketing. So while we shift to Social platforms brands must not forget them. Understanding their needs, addressing them, inviting them to respond and take part in the brand’s online activities while keeping it interesting and entertaining for the customer is the key.
Founder and CEO of Glossier, Emily Weiss, described the brand as, “the first socially-driven beauty brand.” Glossier was certainly the pioneer in the term ‘Instagram brands’ – using the platform not only to build awareness but also to have focused conversation with the customer. Glossier often crowdsources product development, asking its Instagram followers what they’d like to see next. 
Dynamic
The content creation must be dynamic and suitable for the current season, festival, or event that is happening. People must be able to relate to the posts and they must get excited after seeing it.
Starbucks does a marvelous job when it comes to being dynamic. Their one-off recipes and continuous hype around seasonal events and related drinks have won them a silver IPA Effectiveness award for its social strategy in 2018. The launch of its now iconic ‘Unicorn Frappuccino’, spurred on the trend of brands deliberately creating ‘Instagrammable’ food and drink. As a result, there are now 557,232 posts using the hashtag #PumpkinSpiceLatte.
Product focussed
Now of course the real motive behind any marketing activity is to eventually sell the product. So it becomes extremely important to not lose the essence of the product while marketing. In simple terms, the content or strategy must not leave the customer confused about what the brand actually does or what it is trying to sell.
To support the product-centric statement, Oreo is the best example. Oreo’s social media strategy has never diverged from its original formula. Completely product-focused and yet always creative, the brand continuously finds ways to put its cookies center stage. The ‘Daily Twist’ campaign to mark its 100th birthday, saw the brand turn its Oreo cookie into something of cultural relevance for 100 days, including Elvis Presley, a baseball, and a rainbow flag in support of Pride. 
Visuals
“A picture is worth 1,000 words”. This adage has never been more apt than in the age of social media.  This is why visuals are so important in any marketing strategy. You can talk about a product all day, but until you’re able to put it in front of someone’s eyes, it’s not going to have nearly the same effect. Visual content marketing is a great way to make this happen and can be broken down into six basic types- comics, memes, infographics, photos, videos and visual note-taking.
Coca-Cola has been a leader and trendsetter in the visual content marketing space for years. The 2020 initiative video paved the way for multiway communication, as opposed to the single way process that was prevalent. 
Product demonstration:
The barrier for electronic platforms has always been the loss of touch. Customers often complain that till they don’t touch, try or test the product they are not convinced if they want to purchase it. This can also be true for products that are new and the customers need to be educated about its use. Hence it becomes important for brands to bridge this gap as far as possible through their online medium.
Product demonstrations are considered one of the most promising and upcoming applications of AR technology, giving brands a way to provide an immersive experience to consumers. Gucci partnered with Snapchat for the first global branded AR shoe try on the lens, hugely increasing fanfare!
Social message
Corporate Social Responsibility should not be missed when it comes to social media engagement. If a brand has the reach to spread awareness then they must use it. Creating content to spread social awareness adds an emotional touch too. If a customer connects with a brand on an emotional level its hard for competitors to break it.
Dove’s steady and impactful social message still stands out as marketing that’s more than just marketing. The ‘Real Beauty’ campaign has been refreshed multiple times since it first launched in 2004. Dove’s latest campaign once again highlighted the core message, going far beyond surface aesthetic to focus on the beauty of real human values. The ‘Courage is Beautiful’ campaign successfully honored the healthcare workers who have been working selflessly throughout the 2020 pandemic.
The fact remains that the share of mind and voice are key elements to being the dominant force in the world today. Social media marketing has leveraged all the elements and made its presence felt in every customer engagement. Finding unique ways to really stand out among the competition is key to a successful digital marketing strategy. Newer and more customized tools to feel the pulse of the market come up every single day, to meet the ever-changing needs of marketing heads. At the end of the day, ‘customer is indeed king’. And companies would do well to remember it! 
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
