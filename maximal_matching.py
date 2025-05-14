import collections
import math
import pickle
import os
import re
import unicodedata

# First, we will train.

def normalize(string):
    alphabet = 'aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệfghiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvwxyỳỷỹýỵz'
    return re.sub(f'[^{alphabet} ]', '', unicodedata.normalize('NFC', string).lower())

with open('Viet74K.txt', 'r') as file:
    dictionary = {normalize(line) for line in file}

os.chdir('EVBCorpus_EVBNews_v2.0')

# Now, we will test.

true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

for test_number in range(750, 751):
    print(f'Testing... {(test_number - 750) / (1001 - 750):5.2%}')

    with open(f'N{test_number:04d}.sgml', 'r') as file:
        for line in file:
            if not line.startswith("<spair"):
                continue

            next(file)
            syllables = normalize(re.sub(r'<.*?>|</s>', '', next(file))).split()

            # Here, we will get the actual word boundaries from the training data.

            try:
                actual_boundaries = [1 for _ in range(len(syllables) - 1)]
                annotations = re.sub(r'<.*?>|;</a>\n', '', next(file)).split(';')

                for annotation in annotations:
                    syllable_indices = [int(_) - 1 for _ in annotation.split('-')[1].split(',')]

                    for i, j in zip(syllable_indices[:-1], syllable_indices[1:]):
                        if i + 1 == j:
                            actual_boundaries[i] = 0

            except IndexError:
                continue

            except ValueError:
                continue

            # Here, we posit words based on choosing the largest possible words from a dictionary.

            posited_boundaries = [0 for _ in range(len(syllables) - 1)]
            posited_words = []
            word = ''

            for i in range(len(syllables)):
                if word != '' and ' '.join([word, syllables[i]]).strip() not in dictionary:
                    posited_boundaries[i - 1] = 1
                    posited_words.append(word)
                    word = ''

                word = ' '.join([word, syllables[i]]).strip()

            # Here, we will count true/false positives/negatives.

            for i in range(len(syllables) - 1):
                if actual_boundaries[i] == posited_boundaries[i]:
                    if posited_boundaries[i] == 1:
                        true_positives += 1

                    else:
                        true_negatives += 1

                else:
                    if posited_boundaries[i] == 1:
                        false_positives += 1

                    else:
                        false_negatives += 1

            print(syllables)
            # print(words)
            print(posited_words)
            # print(actual_boundaries)
            # print(posited_boundaries)
            #
            # print()

# Here, we will calculate the accuracy, precision, recall, and F1.

accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = (2 * precision * recall) / (precision + recall)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1:', f1)

