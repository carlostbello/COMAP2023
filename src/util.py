import urllib 
import pandas as pd
import joblib
import datetime

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

def predict_word(word : str):
    # Calculate word_score, word_occurence, vowels, repeats
    ws = word_score(word)
    wo = occurrence_score(word)
    v = count_vowels(word)
    r = count_repeats(word)
    data_dict = {'word' : [word], 'word_score' : [ws],  'word_occurrence' : [wo],  'vowels' : [v], 'repeats' : [r]}
    df = pd.DataFrame(data=data_dict)
    # Feed into Random Forest to get Avg Guesses
    df_rf = rf_avg_guesses(df)
    df_lr = lr_avg_guesses(df)
    # Feed into Random Forest to get Distribution
    df_rf = rf_guesses_distribution(df_rf)
    df_lr = rf_guesses_distribution(df_lr)
    return df_rf, df_lr

def rf_avg_guesses(df : pd.DataFrame):
    features = ['word_score',  'word_occurrence',  'vowels', 'repeats']
    targets = 'avg_num_guesses'
    # Load the grid search model
    grid_search = joblib.load('../src/avg_guesses_model.joblib')

    # Use the model to make predictions
    df[targets] = grid_search.predict(df[features])
    return df


def lr_avg_guesses(df: pd.DataFrame):
    features = ['word_score',  'word_occurrence',  'vowels', 'repeats']
    targets = 'avg_num_guesses'
    # Load the grid search model
    grid_search = joblib.load('../src/avg_guesses_model_lr.joblib')

    # Use the model to make predictions
    df[targets] = grid_search.predict(df[features])
    return df

def rf_guesses_distribution(df: pd.DataFrame):
    features = ['avg_num_guesses']
    targets = ['in1', 'in2', 'in3', 'in4',
               'in5', 'in6', 'over6']
    # Load the grid search model
    grid_search = joblib.load('../src/guesses_distribution_model.joblib')

    # Use the model to make predictions
    df[targets] = grid_search.predict(df[features])
    return df

    
if __name__ == '__main__':
    print(predict_word("eerie"))
