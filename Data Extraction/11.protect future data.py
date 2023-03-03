
import requests
from bs4 import BeautifulSoup

# Example URL
url = "https://insights.blackcoffer.com/how-to-protect-future-data-and-its-privacy-blackcoffer/"
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
text = ('''How to protect future data and its privacy?
Before the internet, information was in some ways restricted and more centralized. The only mediums of information were books, newspapers, and word of mouth, etc. But now with the advent of the internet and improvements to computer technology (Moore’s Law), information and data skyrocketed, and it has become this open-system, where information can be distributed to people without any kind of limits.
Various publicly available tools have taken the rocket science out of encrypting (and decrypting) email and files. Data encryption isn’t just for technology geeks; modern tools make it possible for anyone to encrypt emails and other information. “Encryption used to be the sole province of geeks and mathematicians, but a lot has changed in recent years. In particular, various publicly available tools have taken the rocket science out of encrypting (and decrypting) email and files. GPG for Mail, for example, is an open-source plug-in for the Apple Mail program that makes it easy to encrypt, decrypt, sign and verify emails using the OpenPGP standard. And for protecting files, newer versions of Apple’s OS X operating system come with FileVault, a program that encrypts the hard drive of a computer. Those running Microsoft Windows have a similar program. This software will scramble your data, but won’t protect you from government authorities demanding your encryption key under the Regulation of Investigatory Powers Act (2000), which is why some aficionados recommend TrueCrypt, a program with some very interesting facilities, which might have been useful to David Miranda,” explains John Naughton in an article for The Guardian.
One of the most basic, yet often overlooked, data protection tips is backing up your data. Basically, this creates a duplicate copy of your data so that if a device is lost, stolen, or compromised, you don’t also lose your important information. As the U.S. Chamber of Commerce and insurance company Nationwide points out, “According to Nationwide, 68% of small businesses don’t have a disaster recovery plan. The problem with this is the longer it takes you to restore your data, the more money you’ll lose. Gartner found that this downtime can cost companies as much as $300,000 an hour.” 
While you should use sound security practices when you’re making use of the cloud, it can provide an ideal solution for backing up your data. Since data is not stored on a local device, it’s easily accessible even when your hardware becomes compromised. “Cloud storage, where data is kept offsite by a provider, is a guarantee of adequate disaster recovery,” according to this post on TechRadar. Twitter: @techradar
Scammers are sneaky: sometimes malware is cleverly disguised as an email from a friend or a useful website. Malware is a serious issue plaguing many computer users, and it’s known for cropping up in inconspicuous places, unbeknownst to users. Anti-malware protection is essential for laying a foundation of security for your devices. “Malware (short for malicious software) is software designed to infiltrate or damage a computer without your consent. Malware includes computer viruses, worms, trojan horses, spyware, scareware, and more. It can be present on websites and emails or hidden in downloadable files, photos, videos, freeware, or shareware. (However, it should be noted that most websites, shareware, or freeware applications do not come with malware.) The best way to avoid getting infected is to run a good anti-virus protection program, do periodic scans for spyware, avoid clicking on suspicious email links or websites. But scammers are sneaky: sometimes malware is cleverly disguised as an email from a friend or a useful website. Even the most cautious of web-surfers will likely pick up an infection at some point.,” explains Clark Howard. Twitter: @ClarkHoward
Much information can be gleaned through old computing devices, but you can protect your personal data by making hard drives unreadable before disposing of them. “Make old computers’ hard drives unreadable. After you back up your data and transfer the files elsewhere, you should sanitize by disk shredding, magnetically cleaning the disk, or using software to wipe the disk clean. Destroy old computer disks and backup tapes,” according to the Florida Office of the Attorney General. Twitter: @AGPamBondi
Operating system updates are a gigantic pain for users; it’s the honest truth. But they’re a necessary evil, as these updates contain critical security patches that will protect your computer from recently discovered threats. Failing to install these updates means your computer is at risk. “No matter which operating system you use, it’s important that you update it regularly. Windows operating systems are typically updated at least monthly, typically on so-called ‘Patch Tuesday.’ Other operating systems may not be updated quite as frequently or on a regular schedule. It’s best to set your operating system to update automatically. The method for doing so will vary depending upon your particular operating system,” says PrivacyRights.org. Twitter: @PrivacyToday
Many software programs will automatically connect and update to defend against known risks.
In order to ensure that you’re downloading the latest security updates from operating systems and other software, enable automatic updates. “Many software programs will automatically connect and update to defend against known risks. Turn on automatic updates if that’s an available option,” suggests StaySafeOnline.org. Twitter: @StaySafeOnline
A valuable tip for both small business owners and individuals or families, it’s always recommended to secure your wireless network with a password. This prevents unauthorized individuals within proximity to hijack your wireless network. Even if they’re merely attempting to get free Wi-Fi access, you don’t want to inadvertently share private information with other people who are using your network without permission. “If you have a Wi-Fi network for your workplace, make sure it is secure, encrypted, and hidden. To hide your Wi-Fi network, set up your wireless access point or router so it does not broadcast the network name, known as the Service Set Identifier (SSID). Password protect access to the router,” says FCC.gov in an article offering data protection tips for small businesses. Twitter: @FCC
When you’re finished using your computer or laptop, power it off. Leaving computing devices on, and most often, connected to the Internet, opens the door for rogue attacks. “Leaving your computer connected to the Internet when it’s not in use gives scammers 24/7 access to install malware and commit cybercrimes. To be safe, turn off your computer when it’s not in use,” suggests CSID, a division of Experian. Twitter: @ExperianPS_NA

Firewalls assist in blocking dangerous programs, viruses, or spyware before they infiltrate your system.”Firewalls assist in blocking dangerous programs, viruses, or spyware before they infiltrate your system. Various software companies offer firewall protection, but hardware-based firewalls, like those frequently built into network routers, provide a better level of security,” says Geek Squad. Twitter: @GeekSquad
Indiana University Information Technology recommends following the Principle of Least Privilege (PoLP): “Do not log into a computer with administrator rights unless you must do so to perform specific tasks. Running your computer as an administrator (or as a Power User in Windows) leaves your computer vulnerable to security risks and exploits. Simply visiting an unfamiliar Internet site with these high-privilege accounts can cause extreme damage to your computer, such as reformatting your hard drive, deleting all your files, and creating a new user account with administrative access. When you do need to perform tasks as an administrator, always follow security procedures.” Twitter: @IndianaUniv
What’s the difference? “…we recommend you use passphrases–a series of random words or a sentence. The more characters your passphrase has, the stronger it is.  The advantage is these are much easier to remember and type, but still hard for cyber attackers to hack.” explains SANS. Twitter: @SANSAwareness
Encrypt your SIM card in case your phone is ever stolen, or take it out if you are selling your old cell phone. Encrypting your data on your removable storage devices can make it more difficult (albeit not impossible) for criminals to interpret your personal data should your device become lost or stolen. USB drives and SIM cards are excellent examples of removable storage devices that can simply be plugged into another device, enabling the user to access all the data stored on it. Unless, of course, it’s encrypted. “Your USB drive could easily be stolen and put into another computer, where they can steal all of your files and even install malware or viruses onto your flash drive that will infect any computer it is plugged in to. Encrypt your SIM card in case your phone is ever stolen, or take it out if you are selling your old cell phone,” according to Mike Juba in an article on Business2Community. Twitter: @EZSolutionCorp
A Post-It note stuck to the outside of your laptop or tablet is “akin to leaving your keys in your car,” says The Ohio State University’s Office of the Chief Information Officer. Likewise, you shouldn’t leave your laptop in your car. It’s a magnet for identity thieves. Twitter: @OhioState

If you don’t really need your files to be visible to other machines, disable file and media sharing completely. If you have a home wireless network with multiple devices connected, you might find it convenient to share files between machines. However, there’s no reason to make files publicly available if it’s not necessary. “Make sure that you share some of your folders only on the home network. If you don’t really need your files to be visible to other machines, disable file and media sharing completely,” says Kaspersky. Twitter: @kaspersky
HowToGeek offers a series of articles with tips, tricks, and tools for encrypting files or sets of files using various programs and tools. This article covers a method for creating an encrypted volume to easily transport private, sensitive data for access on multiple computers. Twitter: @howtogeeksite
Deleting your information on a computing device rarely means it’s truly deleted permanently. Often, this data still exists on disk and can be recovered by someone who knows what they’re doing (such as, say, a savvy criminal determined to find your personal information). The only way to really ensure that your old data is gone forever is to overwrite it. Luckily, there are tools to streamline this process. PCWorld covers a tool and process for overwriting old data on Windows operating systems. Twitter: @pcworld
If you back up your files to the cloud, remember that even though you delete them on your computer or mobile device, they’re still stored in your cloud account. If you’re diligent about backing up your data and use a secure cloud storage service to do so, you’re headed in the right direction. That said, cloud backups, and any data backups really, create an added step when it comes to deleting old information. Don’t forget to delete files from your backup services in addition to those you remove (or overwrite) on your local devices. “If you back up your files to the cloud, remember that even though you delete them on your computer or mobile device, they’re still stored in your cloud account. To completely delete the file, you’ll also need to remove it from your backup cloud account,” says re/code. Twitter: @Recode
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

