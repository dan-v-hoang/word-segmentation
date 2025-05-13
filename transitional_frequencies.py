import collections
import pickle
import os
import re
import unicodedata

# First, we will train.

def normalize(string):
    alphabet = 'aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệfghiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvwxyỳỷỹýỵz'
    return re.sub(f'[^{alphabet} ]', '', unicodedata.normalize('NFC', string).lower())

training_filename = 'vie_news_2022_1M/vie_news_2022_1M-sentences.txt'
cache_filename = 'transitional_frequencies.pickle'

if cache_filename not in os.listdir():
    with open(training_filename, 'r') as file:
        n = sum(1 for line in file)

    with open(training_filename, 'r') as file:
        frequencies = collections.defaultdict(int)
        transitional_frequencies = collections.defaultdict(int)

        for i, line in enumerate(file):
            print(f'Counting transitional frequencies... {i / n : 5.2%}')
            syllables = normalize(line).split()

            for syllable in syllables:
                frequencies[syllable] += 1

            for i in range(len(syllables) - 1):
                transitional_frequencies[(syllables[i], syllables[i + 1])] += 1

    n = len(transitional_frequencies.values())
    s = sum(transitional_frequencies.values())

    for i, (key, value) in enumerate(transitional_frequencies.items()):
        print(f'Normalizing transitional frequencies... {i / n : 5.2%}')
        transitional_frequencies[key] = 1000000 * value / s

    with open(cache_filename, 'wb') as file:
            pickle.dump((frequencies, transitional_frequencies), file)

with open(cache_filename, 'rb') as file:
    frequencies, transitional_frequencies = pickle.load(file)

# Now, we will test.

os.chdir('EVBCorpus_EVBNews_v2.0')

def test(filenames, threshold, loners, verbose=False):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i, filename in enumerate(filenames):
        print(f'Testing... {i / len(filenames) : 5.2%}')

        with open(filename, 'r') as file:
            for line in file:
                if not line.startswith("<spair"):
                    continue

                if verbose:
                    print(re.findall(r'\d+', line)[0])

                next(file)
                syllables = normalize(re.sub(r'<.*?>|</s>', '', next(file))).split()

                # Here, we will get the actual word boundaries from the test data.

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

                # Here, we will posit word boundaries using transitional frequencies.

                posited_boundaries = [0 for _ in range(len(syllables) - 1)]

                for i in range(len(syllables) - 1):
                    if transitional_frequencies[(syllables[i], syllables[i + 1])] < threshold or syllables[i] in loners or syllables[i + 1] in loners:
                        posited_boundaries[i] = 1

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

                # Here, we will print out the sentence with the word boundaries visualized.

                if verbose:
                    for i, syllable in enumerate(syllables):
                        print(syllable, end=' ')

                        if i < len(syllables) - 1:
                            print(end='| ' if actual_boundaries[i] else '')


                    print()

                    for i, syllable in enumerate(syllables):
                        print(syllable, end=' ')

                        if i < len(syllables) - 1:
                            print(end='| ' if posited_boundaries[i] else '')

                    print('\n')

    # Here, we will calculate the accuracy, precision, recall, and F1.

    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = (2 * precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1

# Here, we will learn the threshold parameter.

best_f1 = 0
best_threshold = 1
loners = set()
loners = {'và', 'của', 'có', 'là', 'trong', 'các', 'với', 'được', 'cho', 'không', 'đã', 'người', 'một', 'công', 'để', 'năm', 'khi', 'những', 'này', 'đến', 'ở', 'đó', 'từ', 'tại', 'nhiều', 'cũng', 'sẽ', 'về', 'vào', 'ra', 'nhà', 'trên'}

withheld = {
    'N0001.sgml',
    'N0002.sgml',
    'N0003.sgml',
    'N0004.sgml',
    'N0005.sgml',
    'N0006.sgml',
    'N0007.sgml',
    'N0008.sgml',
    'N0009.sgml',
    'N0010.sgml'
}

for threshold in range(1, 1000):
    print(f'Trying out some parameters... {threshold / 1000 : 5.2%}')
    accuracy, precision, recall, f1 = test(withheld, threshold, loners)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Here, we will actually test.

accuracy, precision, recall, f1 = test(set(os.listdir()) - withheld, best_threshold, loners)
print('Threshold:', best_threshold)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1:', f1)

# test({'N0777.sgml'}, best_threshold, loners, verbose=True)
