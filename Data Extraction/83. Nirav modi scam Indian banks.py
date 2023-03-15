
import requests
from bs4 import BeautifulSoup

# Example URL
url = 'https://insights.blackcoffer.com/why-scams-like-nirav-modi-happen-with-indian-banks/'
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
text = ('''Why scams like Nirav Modi Happen with Indian banks?
India has been beset by financial scams since Independence. Every 4-5 years, Indian citizens hear about so many trillions of public money being misappropriated and eventually lost by people who the public is supposed to start. In this article, with examples, I would like to point out that there are two factors to scams like Nirav Modi: the complicity of actors within the financial industry and the systemic loopholes in an emerging financial system that has been used for scams like Nirav Modi.
The first financial scam of Independent India was found in 1958 when Feroze Gandhi, MP found that In 1957, Haridas Mundhra got the government-owned Life Insurance Corporation (LIC) to invest Rs. 12.4 million (about US$3.2 million at the time) in the shares of six troubled companies of whom Mundhra held a large number of shares which he was trying to boost by rigging the market. The investment was done under governmental pressure and bypassed the LIC’s investment committee, which was informed of this decision only after the deal had gone through. In the event, LIC lost most of the money. Several leading stockbrokers who were on the LIC Investment Committee testified that the investment could not have been made for the purpose of propping up the market, as was claimed by the Finance Ministry, and that had the LIC consulted the Investment Committee, they would have pointed out Mundhra’s forged shares episode from 1956. The Finance Minister T. T. Krishnamachari, in his testimony, tried to distance himself from the LIC decision, implying that it may have been taken by the Finance Secretary, but Justice M.C. Chagla held that the Minister is constitutionally responsible for the action taken by his secretary and he disowns his actions. Eventually, Krishanamachari had to resign. The Nehru government suffered considerable loss of prestige in the incident.
The 1992 Indian stock market scam was a stock market scam orchestrated by Harshad Mehta. The scam took place in Mumbai and was the biggest market scam of India. The 1992 scam was a systematic stock fraud using bank receipts and stamp paper which caused the Indian Stock market to crash. The scam lead to a complete structural change of the security system of India and introduced a completely new system of stock transactions. The 1992 scam exposed the Indian financial systems through the inherent loopholes of the system. This scam led to the reform of the security system of India and introduced online security systems. The scam was orchestrated in such a way that Mehta secured securities from the State Bank of India against forged cheques signed by corrupt officials and failed to deliver the securities. Mehta made the prices of the stocks soar high through fictitious practices and would go on to sell the stocks that he owned in these companies. The 1992 scam raises many questions which involved many bank officials responsible for collusion with Mehta. The 1992 scam caused an investigation through which many officials were implicated in fraudulent charges. The security system of India took a rapid reform in its fundamental structure post the 1992 scam. The first major structural change was the formation of the National Stock Exchange of India (NSE). The first major reform in the financial sector of India was the formation of the CII Code for Desirable Corporate Governance developed by Rahul Bajaj. Post-1992, a new regulatory board known as the Securities and Exchange Board of India was formed to monitor the National Stock Exchange and the National Securities Depository. There were structural changes in the equity market. The government introduced ten acts of parliament and one constitutional amendment based upon the principles of economic reform and legislative change for the equity market. The NSE introduced online trading in 1994 which changed the dynamics of stock buying and selling. The capital market now opened up nationally as opposed to being confined in Mumbai. The exchange system started functioning based on satellite communications that abolished geographical barriers.
In 2001, Ketan Parekh purchased large stakes in less known small market capitalization companies and jacked up their prices through circular trading with other traders, and collusion with these companies and large institutional investors. This resulted in steep hikes in share prices. It later transpired that promoters and industrialists often gave Parekh funds to artificially rig up their share prices. This set of ten stocks was colloquially referred to as “K-10” stocks and Parekh was playfully referred to as “Pentafour”. The RBI commenced an investigation against Parekh. Around the same time, a bear cartel of brokers in Mumbai opposed to Parekh tried to dump their shares of K-10 stocks. Panicking, Parekh sold off his entire ownership of the so-called K-10 stocks that he had successfully jacked up over the past two years. This resulted in a stock market crash the next day, resulting in large-scale losses for large institutional investors, including insurance companies and mutual funds. A 30 member Joint Parliamentary Committee (JPC) investigation ensued which found that Parekh had been involved in circular trading throughout the time period from and with a variety of companies.
When the Securities Exchange Board of India (SEBI) started scanning an entire spectrum of IPOs launched over 2003, 2004, and 2005, it ended digging up more dirt and probably prevented a larger conspiracy to hijack the market. It involved the manipulation of the primary market—read initial public offers (IPOs)—by financiers and market players by using fictitious or benaami Demat accounts. They then transferred the shares to financiers, who sold on the first day of listing, making windfall gains from the price difference between the IPO price and the listing price. This time, fraudsters targeted the primary market to make a quick buck at the expense of the gullible small investors. Direct Participants (DPs) used retail applicants’ shares for reaping benefits in the stock market.
The 2010 fake housing loan in India was uncovered by the Central Bureau of Investigation (CBI) in India. CBI arrested eight top-ranking officials of public sector banks and financial institutions, including the LIC Housing Finance CEO Ramchandran Nair, in connection with the scam. CBI alleged that the officers of various public sector banks and financial institutions received bribes from a private financial services company, which acted as a mediator for corporate loans and other facilities from financial institutions. The bank officials sanctioned large-scale corporate loans to realty developers, overriding mandatory conditions for such approvals along with other irregularities.
The Thane police, struggling to crack down on the tax refund scam that has taken the income tax department for a ride and defrauded the central exchequer of around Rs 3 crore, has concluded that the scam involved high-ranking officials from the I-T department and the State Bank of India. The fraud-involving two I-T offices in Thane and Kalyan was planned by those familiar with the I-T office functioning. For over two years, 2007-2008 and 2008-2009, the I-T officials disbursed tax refund cheques drawn from the SBI and credited into the account of fictitious assessees in a credit society in Dombivli. As many as 331 such tax refund cheques, some exceeding Rs 1 lakh, were subsequently routed to the Dombivli Nagri Cooperative Bank and funds withdrawn. The I-T refunds are account payee cheques, which can’t be transferred. Despite this, all 331 refund orders were credited to the third-party account of a credit society.
The Saradha Group financial scandal was a major financial scam and alleged political scandal caused by the collapse of a Ponzi scheme run by Saradha Group, a consortium of over 200 private companies that were believed to be running collective investment schemes popularly but incorrectly referred to as chit funds in Eastern India. It was feared that legitimate non-banking financial companies and microfinance institutions would be stigmatized, leading to a vicious cycle of low depositor trust, higher interest rates, lower lending and a localized credit crisis. Because most of the Saradha Group depositors came from the lowest economic strata, the loss of the investment would cause a further decrease in social mobility. The scandal drew attention to similar illegal deposit mobilizing companies, which are facing increased regulatory pressure. Many of these companies have been variations of timeshare travel schemes, of which there are few clear regulations. In August 2013, the Central Government amended the SEBI Act and gave SEBI powers to search and seize without prior magisterial permission to investigate illegal money collection schemes.
NSEL case relates to a payment default at the National Spot Exchange Limited that occurred in 2013 involving Financial Technologies India Ltd when a payment default took place after a commodities market regulator, the Forward Markets Commission (FMC), directed NSEL to stop launching contracts. This led to the closure of the Exchange in July 2013. Three spot exchanges, NSEL, NSPOT, and National APMC were exempted by the government under Section 27 of FCRA to conduct forward trading in one-day contracts. This was done to boost volumes so that their economic viability improved. On the flawed recommendations of the FMC, the Ministry of Consumer Affairs ordered NSEL to settle all existing contracts and not launch any fresh contracts, which led to the crisis. Investigations led by Enforcement Directorate (ED) and Economic Offences Wing (EOW) revealed the role of brokers and defaulters in the NSEL case. The brokers mis-sold NSEL products to their clients by assuring them of fixed returns. The defaulters hypothecated stocks and produced fake warehouse receipts and siphoned the entire default money. FIU (under Finance Ministry) held that NSEL came under the purview of the Forwarding Contracts (Regulation) Act (FCRA) and therefore guilty of failing in several of these obligations under the law. The black money watchdog has slapped a penalty of Rs 1.66 crore for several counts of violating the provisions of the Prevention of Money Laundering Act (PMLA) on NSEL. The watchdog further held that failures are deliberate and willful and hence, invite penalties.
The NSE co-location scam relates to the market manipulation at the National Stock Exchange of India, India’s leading stock exchange. Allegedly select players obtained market price information ahead of the rest of the market, enabling them to front-run the rest of the market, possibly breaching the NSE’s purpose of demutualization exchange governance and its robust transparency-based mechanism. he alleged connivance of insiders by rigging NSE’s algo-trading and use of co-location servers ensured substantial profits to a set of brokers. The whistle-blower alleged that trading members were able to capitalize on advanced knowledge by colluding with some exchange officials. In a written reply, Minister of State for Finance, Arjun Meghwal told the Lok Sabha in 2016: “The architecture of NSE with respect to dissemination of tick-by-tick through transmission control protocol (TCP) or internet protocol (IP) was prone to manipulation or market abuse.
The Punjab National Bank Fraud Case relates to the fraudulent letters of undertaking worth ₹11,356.84 crores (US$ 1.4 billion) issued by the Punjab National Bank at its Brady House branch in Fort, Mumbai; making Punjab National Bank liable for the amount. The fraud was allegedly organized by jeweler and designer Nirav Modi. Nirav, his wife Ami Modi, brother Nishal Modi and uncle Mehul Choksi, all partners of the firms, M/s Diamond R US, M/s Solar Exports and M/s Stellar Diamonds; along with PNB officials and employees, and directors of Nirav Modi and Mehul Choksi’s firms have all been named in a charge sheet by the CBI. The bank initially said that two of its employees at the branch were involved in the scam, as the bank’s core banking system was bypassed when the corrupt employees issued LOUs to overseas branches of other Indian banks, including Allahabad Bank, Axis Bank, and Union Bank of India, using the international financial communication system, SWIFT. On 1 March 2018, the government approved the Fugitive Economic Offenders Bill to deter economic offenders from evading the process of Indian law by giving powers to the government to confiscate assets of a fugitive, including Benami assets of absconding loan defaulters. The bill covers a wide range of economic offenders which include: loan defaulters, fraudsters, individuals who violate laws governing taxes, black money, Benami properties, the financial sector, and corruption. In March 2018, the Reserve Bank of India scrapped banking instruments such as the Letter of understanding (Lou) and Letter of Comfort (LoC) that in an attempt to plug a loophole and improve banks’ due diligence in trade credit. Some bankers said that LoUs and LoCs led to receiving banks depending completely on the issuing bank on creditworthiness.
As seen in all the above cases for scams like Nirav Modi, the two factors: action taken by the insiders or systemic loopholes were responsible for the majority of the Indian financial scams. The improvement in the industry will only come when all the loopholes are closed down and accountability is given and held to all the people working in the industry.
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

