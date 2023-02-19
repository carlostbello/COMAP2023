import urllib 
import requests
import pandas as pd



allowed = pd.read_csv('../data/allowed_words.csv')
allowed_words = allowed.word.values
freqs = allowed.freqs

def calculate_freqs():
    letter_freq = {}
    for word in allowed_words:
        for char in word:
            if char in letter_freq:
                letter_freq[char] += 1
            else:
                letter_freq[char] = 1

    for key in letter_freq.keys():
        letter_freq[key] /= len(allowed_words*5)
    return letter_freq

letter_freqs = calculate_freqs()

def word_freq(word):
    return sum([letter_freqs[letter] for letter in word])

def word_freqs(words):
    return words.apply(word_freq)

def occurrence_score(word):
    return allowed[allowed.word == word].freqs.values[0]
