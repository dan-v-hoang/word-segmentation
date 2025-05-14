import collections
import math
import pickle
import os
import re
import unicodedata

# First, we will train.

os.chdir('EVBCorpus_EVBNews_v2.0')

def normalize(string):
    alphabet = 'aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệfghiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvwxyỳỷỹýỵz'
    return re.sub(f'[^{alphabet} ]', '', unicodedata.normalize('NFC', string).lower())

unigram_frequencies = collections.defaultdict(int)
bigram_frequencies = collections.defaultdict(int)

for training_number in range(1, 750):
    print(f'Training... {training_number / 750:5.2%}')

    with open(f'N{training_number:04d}.sgml', 'r') as file:
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

            # Here, we get the actual words.

            words = ['']

            for i in range(len(syllables) - 1):
                words[-1] += syllables[i] + ' '

                if actual_boundaries[i]:
                    words[-1] = words[-1].strip()
                    words.append('')

            words[-1] += syllables[-1]

            # Here, we count unigram and bigram frequencies.

            bigram_frequencies[('<bos>', words[0])] += 1
            bigram_frequencies[(words[-1], '<eos>')] += 1

            for i in range(len(words)):
                unigram_frequencies[words[i]] += 1

                if i < len(words) - 1:
                    bigram_frequencies[(words[i], words[i + 1])] += 1

# Here, we will conver the unigram and bigram frequencies to probabilities.

s = sum(unigram_frequencies.values())

for unigram in unigram_frequencies:
    unigram_frequencies[unigram] /= s

s = sum(bigram_frequencies.values())

for bigram in bigram_frequencies:
    bigram_frequencies[bigram] /= s


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

            # Here, we get the actual words.

            words = ['']

            for i in range(len(syllables) - 1):
                words[-1] += syllables[i] + ' '

                if actual_boundaries[i]:
                    words[-1] = words[-1].strip()
                    words.append('')

            words[-1] += syllables[-1]

            # Here, we posit words based on maximizing the product of bigram frequencies.
            # This is done by setting up a graph where
            # the vertices correspond to syllables
            # and the edges correspond to choices of word segmentation.
            # An edge occurs between two vertices if the syllables including and between the vertices forms a valid word.
            # This graph is a dag, so we can use dynamic programming to score paths.

            graph = collections.defaultdict(set)

            for i in range(len(syllables)):
                for j in range(i, len(syllables)):
                    next_word = ' '.join(syllables[i:j + 1])

                    if j + 1 - i > 1 and next_word not in unigram_frequencies:
                        break

                    graph[i].add((i + (j + 1 - i), next_word))

            previous_index = {}
            previous_word = {}
            scores = {}
            previous_word[0] = '<bos>'
            scores[0] = 1
            s = sum(bigram_frequencies.values())

            for i in range(len(syllables)):
                for j, word in graph[i]:
                    score = scores[i] * bigram_frequencies[(previous_word[i], word)]

                    if score == 0:
                        score = scores[i] * (1 / s)

                    if j not in scores or scores[j] < score:
                        previous_index[j] = i
                        previous_word[j] = word
                        scores[j] = score

            # Here, we will do a backward pass to extract the whole segmentation calculated by DP.

            posited_boundaries = [0 for _ in range(len(syllables) - 1)]
            posited_words = []
            i = len(syllables)

            while i != 0:
                if i != len(syllables):
                    posited_boundaries[i - 1] = 1

                posited_words.append(previous_word[i])
                i = previous_index[i]

            posited_words.reverse()

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

            # print(syllables)
            # print(words)
            # print(posited_words)
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

