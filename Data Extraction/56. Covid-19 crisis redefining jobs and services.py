
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/how-the-covid-19-crisis-is-redefining-jobs-and-services/'

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
text = ('''How the COVID-19 crisis is redefining jobs and services?
The pandemic has shaken the world in the way one had hardly imagined. The impact of it on Jobs and Services is going to be long-lasting and it will be the foundation stone of revolution in these sectors. With the work from home and social distancing norms is altering the way of work, the way we interact, what work we will do, how the work will be done, and whatnot. To just give an example, in the middle of the first lockdown, FMCG major Hindustan Unilever (HUL) went ahead with its merger. It took place online entirely. Yes, it took place online. Who could have ever imagined the merger can take place online?
Change is inevitable. But some changes are temporary and some are permanent. The pandemic is bringing changes that are going to be permanent. In fact, these changes were due for a long time and we’re coming at a slow pace. What the pandemic has done is, it has become the catalyst for these changes and raised the ante for change. So, what exactly the changes will be? Let’s look at them one by one.
The first change is going to be the extensive use of digital communication from now onwards. The Zoom, Jio Meet, Google Meet, and some derivatives like these are going to be the forever meeting rooms where the important discussions will be held. This may be bad news for the employees who are older than 40 years. They would not have pictured themselves doing such transformation at this part of life. Experts believe that the new normal in the offices are going to be hierarchy-less. The older employees would have to work with the younger ones. There would be no geographical boundaries and where the communication and software adaptability among employees is going to be crucial. The symbol of power will be less visible and the decentralization of power in the office will be prominent. There will be more egalitarianism in companies. The corporates will think much more about office space. Rather than expending too much on physical infrastructure, the online infrastructure will always be in their hindsight. The impact would be more prudent in small and mid-sized businesses. The physical offices itself might disappear from such businesses. But it’s not going to be easy for companies. The security and confidentiality of the authentic information going out of the offices’ building are going to be the largest concern in the new normal world. There would be no surprise if a lot of start-ups will bloom in the near future providing software for security for the data and information of the companies. Also, travel is going to be reduced. So, the major component of CTC where the companies fooling the employees must find the other way.
The second change is going to be struck down on employees. Already many of the companies and start-ups are reducing the workforce. It is expected that new normal offices would work with an 80% workforce of that of the current workforce. Automation is on the verge. Due to safety reasons, the companies are going to be more and more leaned towards automation with a minimal touch of human possible. This might result in a loss of jobs in some sectors, but it will also create new jobs in the automation sector. It was evident from before pandemic too but now the velocity for the same would be increased. In the new normal tech-enabled world, the job which would be at stake dominantly would be managers’ managers. The IT firms have started firing the vice presidents and assistant vice presidents who draw hefty salaries but don’t really drive the business. In the designation-less set up in the near future accelerated by Pandemic, they automatically become redundant. For the first time, the employees might be given jobs that would require more than the standard jobs. Also, when the workforce in the company would be less in the company, the employees might be given the option of working for extra time or working on the weekends and would be incentivized for the same rather than employing more staff and paying them full salary. It is a win-win situation for both employees and companies. Employees can earn more than what they usually earn and will get to learn new which would be out of their primary domain. The companies would be able to keep their costs low through this. The assessment of employees might change too. Rather than going for an annual performance review, the employees might be evaluated based on the tasks they perform. Creativity and innovation are going to be swords for the employees in the new normal world rather than designations in the office.  
The other change might be in the services sector. The services like which involves maximum human touch will be the one which will be affected most adversely. The safety concerns and confidence of people over the system are going to be the issue for at least the next few years. It also opens the opportunity for something very new. The products which could replace the need for visiting such services centers are going to create a boom in the market. According to McKinsey, the sales for such products are multiplying rapidly.
There are speculations that the economies will tend to become Minimalist economies. Let’s dig deep into this. Minimalist economies are the economies where people tend to refrain from purchasing luxurious items and focuses on purchasing necessities more and more.  Well, this is the position we are in right now during the pandemic. People are expensing only necessary items like groceries and the demand for luxurious items is way below average. But it would be hard to stay the same as this. The minimalist economy will result in the deterioration of the economy in the short term. It is true that in the minimalist economy the demand might be low, but it would be wrong to assume the demand will be zero. The high-quality products will gather the attention of the purchasers. It is evident from past experiences that the economy always finds its way to come back from the mode of minimalist economy. Though this time may take some time, it will surely return on track.
But every trouble comes with an opportunity. This is the time for the companies to act now and grab the opportunities the pandemic has brought with him. But for that companies are required to follow plans, formulate strategies to suit their needs, and pay more attention to the innovation. The companies must form plans for the five stages from here. These stages are 5Rs: Resolve, Resilience, Return, Reimagination, and Reform.
Resolve: In the resolve phase companies need to formulate a nerve center to combat Corvid itself. The tendency of the company should not be like that, they will return only after everything becomes normal. In most parts of the country, permission is granted to the companies for their operations. If they wait for normalcy, then bankruptcy might catch them before the normalcy.
Resilience: The resilience phase includes maintaining liquidity, addressing solvency, and grow for sustainability. All businesses should know when their cash crunch is coming. Addressing these cash crunches will be crucial for the companies. Businesses must take aggressive options to remain solvent. For example, the businesses might have cash in their hand but might be poor in operational efficiency then it requires the attention of the company. Organizations solving issues of liquidity and solvency will be in a better position to grow with sustainability.
Return: In the third phase of return the companies will have to plan for the time when everything will be back on track. Given the possibility of subsequent waves of coronavirus the companies need new ways of working to prevent, identify, report, and contain future flareups. Many industries will face this problem of returning. Staying prepared for resurgence scenarios should include a multi-scenario modeling exercise. Reverting to non-COVID-19 care will require extensive planning and market testing.
Reimagine: The phase of reimagining will include the learnings from the pandemic. A small virus and the whole economy at the toss. The businesses will have to introspect them and must look for advancements in technologies such that this kind of pandemic in the future will not disturb their efficiency too much. The pandemic has taught us that the decision which took weeks and months in the normal world can be accelerated and can be taken in some days. This should become a permanent feature. Cross organizational collaboration has become much easier. This should stay in the long run.
Reform: The last phase is the reform. In this phase, the companies should look at their bottlenecks and try to fix them. Also, the companies will have to reconsider their relationship with their customers. The confidence in their brands, they must regain it.
There are three ways to shift work, talent, and skills to where and when they are needed most, thereby building the organizational resilience and agility necessary to navigate uncertain times and rebound with strength when the economy recovers.
In this dismal situation it is more important than ever most of the staff should be in the critical tasks. The tasks could be like the customer complaints redressal. We know that the customers are the ones who make the businesses successful. Thus, retaining their confidence is the utmost priority. The organizations can convert some of the staff to address the queries and concerns of customers. The organizations require to create virtual offices. There should not be the geographical boundaries within the organization. The employees from all over the world from the organization should come together and take the organization back to pre-corvid state. By breaking out of rigid job constraints, the right talent and work can be matched to solve evolving business challenges in real-time.
This was happening before the pandemic too. But accelerating it with pace has now become the need of the hour. The perception that automation is a job-killer is totally wrong and in fact, it is the mandatory capability to deal with the crisis. Organizations can increase automation in call centers. This will reduce the response time.
Temporarily moving employees from some industries like airlines, restaurants, hospitality can be moved to those organizations which have maximum work at this time like healthcare and logistics.
To conclude, the companies should understand “Lives come first, but livelihood matters”. The pandemic will have some adverse effects on jobs and services, but this is not the end. The pandemic has brought many opportunities with him and if they are grabbed then the organizations would be in much better position even compared to pre-corvid situations. The companies need to understand the pandemic is not going to be forever. The last concluding sentence will be corona will bring changes but there will not be a revolution in jobs and services.
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

