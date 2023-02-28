
import requests
from bs4 import BeautifulSoup

# Example URL
url = "https://insights.blackcoffer.com/deep-learning-impact-on-areas-of-e-learning/"

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
text = ('''Deep learning impact on areas of e-learning?
eLearning as technology becomes more affordable in higher education but having a big barrier in the cost of developing its resources. Deep learning using artificial intelligence continues to become more and more popular and having impacts on many areas of eLearning. It offers online learners of the future intuitive algorithms and automated delivery of eLearning content through modern LMS platforms. This paper aims to survey various applications of deep learning approaches for developing the resources of the eLearning platform, in which predictions, algorithms, and analytics come together to create more personalized future eLearning experiences. In addition, deep learning models for developing the contents of the eLearning platform, deep learning framework that enable deep learning systems into eLearning and its development, benefits & future trends of deep learning in eLearning, the relevant deep learning-based artificial intelligence tools, and a platform enabling the developer and learners to quickly reuse resources are clearly summarized. Thus, deep learning has evolved into developing ways to re-purpose existing resources that can mitigate the expense of content development of future eLearning.
It is natural to wonder where you might get AI tools to avoid the time and expense of developing your own. Don’t worry about the advert of AIaaS or “AI as a Service” even small education or learning & development professionals can purchase the license of AI tools and components. However, such types of tools cannot be useful for every e-learning ecosystem but may offer some enticing benefits such as adding standard AI tasks (logic, decision making) to your toolbox. Here are some of the AIaaS tools and platforms offered by famous tech giants most of which are cloud-based.
Microsoft Azure
Cloud-based AI services that can be used to build and manage AI applications like image recognition or bot-based apps
IBM’s Watson
Cloud-based AI services that can be integrated into your applications; to store and manage your own data
Google’s Tensor Flow
 An end-to-end open-source machine learning platform
Amazon Web Services
Offers a wide range of products and services on Amazon’s cloud
There are other AIaaS platforms such as DataRobot, Petuum, and H2O which shows that the field is expanding.AI will probably not make human workers obsolete, at least not for a long time To put some of your fears to bed: the robots are probably not coming for your jobs, at least not yet. Given how artificial intelligence has been portrayed in the media, in particular in some of our favorite sci-fi movies, it’s clear that the advent of this technology has created fear that AI will one day make human beings obsolete in the workforce. After all, as technology has advanced, many tasks that were once executed by human hands have become automated. It’s only natural to fear that the leap toward creating intelligent computers could herald the beginning of the end of work as we know it. But, I don’t think there is any reason to be so fatalistic. A recent paper published by the MIT Task Force on the Work of the Future entitled “Artificial Intelligence And The Future of Work,” looked closely at developments in AI and their relation to the world of work. The paper paints a more optimistic picture.
Rather than promoting the obsolescence of human labor, the paper predicts that AI will continue to drive massive innovation that will fuel many existing industries and could have the potential to create many new sectors for growth, ultimately leading to the creation of more jobs. While AI has made major strides toward replicating the efficacy of human intelligence in executing certain tasks, there are still major limitations. In particular, AI programs are typically only capable of “specialized” intelligence, meaning they can solve only one problem, and execute only one task at a time. Often, they can be rigid, and unable to respond to any changes in input, or perform any “thinking” outside of their prescribed programming. Humans, however, possess “generalized intelligence,” with the kind of problem-solving, abstract thinking, and critical judgment that will continue to be important in business. Human judgment will be relevant, if not in every task, then certainly throughout every level across all sectors. There are many other factors that could limit runaway advancement in AI. AI often requires “learning” which can involve massive amounts of data, calling into question the availability of the right kind of data, and highlighting the need for categorization and issues of privacy and security around such data. There is also the limitation of computation and processing power. The cost of electricity alone to power one supercharged language model AI was estimated at $4.6 million. Another important limitation of note is that data can itself carry bias, and be reflective of societal inequities or the implicit biases of the designers who create and input the data. If there is bias in the data that is inputted into an AI, this bias is likely to carry over to the results generated by the AI.
There has even been a bill introduced into Congress entitled the Algorithmic Accountability Act with the goal of forcing the Federal Trade Commission to investigate the use of any new AI technology for the potential to perpetuate bias. Based on these factors and many others, the MIT CCI paper argues that we are a long way from reaching a point in which AI is comparable to human intelligence, and could theoretically replace human workers entirely.  Provided there is an investment at all levels, from education to the private sector and governmental organizations—anywhere that focuses on training and upskilling workers—AI has the potential to ultimately create more jobs, not less. The question should then become not “humans or computers” but “humans and computers” involved in complex systems that advance industry and prosperity. This paper is a fascinating read for anyone hoping to dive deeper into AI and the many potential directions in which it may lead.AI Is becoming standard in all businesses, not just in the world of tech A couple of times recently, AI has come up in conversation with a client or an associate, and I’m noticing a fallacy in how people are thinking about it. There seems to be a sense for many that it is a phenomenon that is only likely to have big impacts in the tech world. In case you hadn’t noticed, the tech world is the world these days. Don’t ever forget when economist Paul Krugman said in 1998 that “By 2005 or so, it will become clear that the Internet’s impact on the economy has been no greater than the fax machine’s.” You definitely don’t want to be behind the curve when it comes to AI.  In fact, 90% of leading businesses already have ongoing investments in AI technologies. More than half of businesses that have implemented some manner of AI-driven technology report experiencing greater productivity. AI is likely to have a strong impact on certain sectors in particular:
Medical:
The potential benefits of utilizing AI in the field of medicine are already being explored. The medical industry has a robust amount of data, which can be utilized to create predictive models related to healthcare. Additionally, AI has shown to be more effective than physicians in certain diagnostic contexts.
Automotive:
We’re already seeing how AI is impacting the world of transportation and automobiles with the advent of autonomous vehicles and autonomous navigation. AI will also have a major impact on manufacturing, including within the automotive sector.
Cybersecurity:
Cybersecurity is front of mind for many business leaders, especially considering the spike in cybersecurity breaches throughout 2020. Attacks rose 600% during the pandemic as hackers capitalized on people working from home, on less secure technological systems, and Wi-Fi networks. AI and machine learning will be critical tools in identifying and predicting threats in cybersecurity. AI will also be a crucial asset for security in the world of finance, given that it can process large amounts of data to predict and catch instances of fraud.
E-Commerce:
AI will play a pivotal role in e-commerce in the future, in every sector of the industry from user experience to marketing to fulfillment and distribution. We can expect that moving forward, AI will continue to drive e-commerce, including through the use of chat-bots, shopper personalization, image-based targeting advertising, and warehouse and inventory automation.
AI can have a big impact on the job search
If you are moving forward with the hope that a hiring manager may give you the benefit of the doubt on a small misstep within the application, you might be in for a rude awakening. AI already plays a major role in the hiring process, so much so that up to 75% of resumes are rejected by an automated applicant tracking system, or ATS before they even reach a human being.  In the past, recruiters have had to devote considerable time to poring over resumes to look for relevant candidates. Data from LinkedIn shows that recruiters can spend up to 23 hours looking over resumes for one successful hire.
Increasingly, however, resume scanning is being done by AI-powered programs. In 2018, 67% of hiring managers stated that AI was making their jobs easier. Despite the increasing prevalence of automation and algorithms in the hiring process, many have been critical of the use of certain types of AI by hiring managers, based on the charge that it can perpetuate and ever create more bias in hiring. One particular example is illustrated by HireVue, a startup whose initial services included technology that aimed to use facial recognition software and psychology to determine the potential effectiveness of a candidate in a certain role. The Electronic Privacy Information Center filed a lawsuit with the Federal Trade Commission alleging that this software had the potential to perpetuate bias and prejudice. HireVue discontinued the use of facial recognition software in early 2021, and now uses audio analysis and natural language processing. It’s clear that the use of certain types of AI in the hiring process will likely be controversial as new technology develops. However, if potential employers are using AI to process your application, there is no reason that you cannot be utilizing similar technology to your advantage.
Jobscan is an excellent resource that provides similar resume scanning to what would be used by a hiring manager. By comparing your resume to a job description, Jobscan will give you information on how to tweak your resume so that it is a good match for a certain position, with the goal of “beating” an applicant tracking system (ATS).
Jobseer is a browser add-on, and another great AI-based tool for those on the job market. Based on a scan of your resume, as well as keywords and skills related to your desired jobs, Jobseer will help match you with the job listings that best fit your experience. For each listing, you get a rating based on how well you are aligned with the particular posting, as well as recommendations of skills to add to better position your resume and experience.
Rezi: Now, as a disclaimer, I would never encourage you to turn your resume writing over to a bot. But Rezi is an awesome AI-based resume builder that includes templates to help you design a resume that is sure to check the boxes when it comes to applicant tracking systems. This is a great jumping-off point to kickstart a new resume.  Another great way to use this type of tool is to generate a new resume and compare it to your current resume to see how it stacks up, and identify some areas for improvement. AI is also a great place to focus your energy if you are looking to upskill in your career, or make your professional profile more competitive in the job market, especially when you consider that AI will have such far-reaching impacts across many industries.AI and machine learning are at the top of many lists of the most important skills in today’s job market. Jobs requesting AI or machine-learning skills are expected to increase by 71% in the next five years. If you’d like to expand your knowledge base in this arena, consider some of the great free online course offerings that focus on AI skills. If you are tech-savvy, it would be wise to dive deep and learn as much as you can about interacting in the AI space. If your skills lie elsewhere, it is important to recognize that AI will have a big impact, and to the extent of your abilities, you should try to understand the fundamentals of how it functions in different sectors. AI is definitely here to stay, whether we like it or not. Personally, I don’t think we have anything to be afraid of. The best way to move forward is to be aware of and adapt to the new technology around us, AI included. This article was updated on April 16, 2021, to reflect changes in HireVue’s assessment tools.
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

