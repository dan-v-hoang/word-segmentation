import collections
import math
import pickle
import os
import re
import unicodedata

# First, we will train. The notation in this file follows Jurafsky and Martin 2023.

def normalize(string):
    alphabet = 'aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệfghiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvwxyỳỷỹýỵz'
    return re.sub(f'[^{alphabet} ]', '', unicodedata.normalize('NFC', string).lower())

Q = {'B', 'I'}
A = {'B': {'B': 0, 'I': 0}, 'I': {'B': 0, 'I': 0}}
B = collections.defaultdict(lambda: collections.defaultdict(int))
π = {'B': 1, 'I': 0}
os.chdir('EVBCorpus_EVBNews_v2.0')
verbose = False
n = 750 # This is where to draw the line between training and test data.

for training_number in range(1, n):
    print(f'Training... {training_number / n : 5.2%}')

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

            # Here, we convert to BI-tags.

            actual_bi_tags = [None for _ in range(len(syllables))]
            actual_bi_tags[0] = 'B'

            for i in range(len(syllables) - 1):
                actual_bi_tags[i + 1] = 'B' if actual_boundaries[i] else 'I'

            # Here, we count conditional frequencies so that we can calculate the A and B matrices later.

            for i in range(len(syllables) - 1):
                A[actual_bi_tags[i]][actual_bi_tags[i + 1]] += 1

            for i in range(len(syllables)):
                B[syllables[i]][actual_bi_tags[i]] += 1

            if verbose:
                print(actual_bi_tags)

                for i, syllable in enumerate(syllables):
                    print(syllable, end=' ')

                    if i < len(syllables) - 1:
                        print(end='| ' if actual_boundaries[i] else '')

                print()

# Here, we will convert A and B from frequencies to probabilities.

for before in A:
    s = sum(A[before].values())

    for after in A[before]:
        A[before][after] /= s

for syllable in B:
    s = sum(B[syllable].values())

    for bi_tag in B[syllable]:
        B[syllable][bi_tag] /= s

# Now, we will test.

true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
verbose = False

for test_number in range(n, 1001):
    print(f'Testing... {(test_number - n) / (1001 - n) : 5.2%}')

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

            # Here, we convert to BI-tags.

            actual_bi_tags = [None for _ in range(len(syllables))]
            actual_bi_tags[0] = 'B'

            for i in range(len(syllables) - 1):
                actual_bi_tags[i + 1] = 'B' if actual_boundaries[i] else 'I'

            # Here, we will posit word boundaries using the HMM we just trained and the Viterbi algorithm.
            # Again, I try to follow the notation in the pseudocode given in Jurafsky and Martin.

            viterbi = {}
            backpointer = {}

            for s in Q:
                viterbi[(s, 0)] = π[s] * B[syllables[0]][s]
                backpointer[(s, 0)] = None

            T = len(syllables)

            for t in range(1, T):
                for s in Q:
                    viterbi[(s, t)], backpointer[(s, t)] = max([(viterbi[(s_0, t - 1)] * A[s_0][s] * B[syllables[t]][s], s_0) for s_0 in Q])

            bestpathprob, bestpathpointer = max([(viterbi[(s, T - 1)], s) for s in Q])
            bestpath = []

            for t in range(T - 1, -1, -1):
                bestpath.append(bestpathpointer)
                bestpathpointer = backpointer[(bestpathpointer, t)]

            bestpath.reverse()
            posited_bi_tags = bestpath

            # Here, we count conditional frequencies so that we can calculate the B matrix later.

            for i in range(len(syllables)):
                B[syllables[i]][actual_bi_tags[i]] += 1

            # Here, we will count true/false positives/negatives.

            for i in range(len(syllables) - 1):
                if actual_bi_tags[i] == posited_bi_tags[i]:
                    if posited_bi_tags[i] == 'B':
                        true_positives += 1

                    else:
                        true_negatives += 1

                else:
                    if posited_bi_tags[i] == 'B':
                        false_positives += 1

                    else:
                        false_negatives += 1

            if verbose:
                print(actual_bi_tags)

                for i, syllable in enumerate(syllables):
                    print(syllable, end=' ')

                    if i < len(syllables) - 1:
                        print(end='| ' if actual_boundaries[i] else '')

                print()
                print(posited_bi_tags)

                for i, syllable in enumerate(syllables):
                    print(syllable, end=' ')

                    if i < len(syllables) - 1:
                        print(end='| ' if posited_bi_tags[i + 1] == 'B' else '')

                print()

accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = (2 * precision * recall) / (precision + recall)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1:', f1)
