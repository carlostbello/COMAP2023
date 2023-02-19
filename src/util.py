import urllib 
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

def word_score(word):
    return sum([letter_freqs[letter] for letter in word])

def word_scores(words):
    return words.apply(word_score)

def occurrence_score(word):
    return allowed[allowed.word == word].freqs.values[0]


vowels = ['a','e','i','o','u']
def count_vowels(word):
    count = 0
    for letter in word:
        if letter in vowels:
            count+=1
    return count

def count_repeats(word):
    letters = set([letter for letter in word])
    return 5-len(letters)