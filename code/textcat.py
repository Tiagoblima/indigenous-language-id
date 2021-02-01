# Ensure that literal strings default to unicode rather than str.
from __future__ import print_function, unicode_literals

import os

from nltk.compat import PY3
from nltk.util import trigrams

from params import CLASSES_

if PY3:
    from sys import maxsize
else:
    from sys import maxint

# Note: this is NOT "re" you're likely used to. The regex module
# is an alternative to the standard re module that supports
# Unicode codepoint properties with the \p{} syntax.
# You may have to "pip install regx"
try:
    import regex as re
except ImportError:
    re = None
from nltk import FreqDist, trigrams

from os import path

from six import PY3


class CrubadanCorpusReader():
    """
    A corpus reader used to access language An Crubadan n-gram files.
    """

    _all_lang_freq = {}

    def __init__(self, root='profiles/dataset1', encoding='utf8', tagset=None):
        # super(CrubadanCorpusReader, self).__init__(root, fileids, encoding='utf8')
        self._all_lang_freq = {}
        self.root_dir = root
        # self.fileids = fileids
        self._lang_mapping_data = []

    def lang_freq(self, lang):
        """ Return n-gram FreqDist for a specific language
            given ISO 639-3 language code """

        if lang not in self._all_lang_freq:
            self._all_lang_freq[lang] = self._load_lang_ngrams(lang)

        return self._all_lang_freq[lang]

    def langs(self):
        """ Return a list of supported languages as ISO 639-3 codes """
        return CLASSES_

    def iso_to_crubadan(self, lang):
        """ Return internal Crubadan code based on ISO 639-3 code """
        for i in self._lang_mapping_data:
            if i[1].lower() == lang.lower():
                return i[0]

    def crubadan_to_iso(self, lang):
        """ Return ISO 639-3 code given internal Crubadan code """
        for i in self._lang_mapping_data:
            if i[0].lower() == lang.lower():
                return i[1]

    def _load_lang_ngrams(self, lang):
        """ Load single n-gram language file given the ISO 639-3 language code
            and return its FreqDist """

        ngram_file = path.join(self.root_dir, lang + '.txt')

        if not path.isfile(ngram_file):
            raise RuntimeError("No N-gram file found for requested language.")

        counts = FreqDist()
        if PY3:
            f = open(ngram_file, 'r', encoding='utf-8')
        else:
            f = open(ngram_file, 'rU')

        for line in f:
            if PY3:
                data = line.split(' ')
            else:
                data = line.decode('utf8').split(' ')

            ngram = data[0]
            freq = int(data[1].strip('\n'))

            counts[ngram] = freq

        return counts


class TextCat(object):
    _corpus = None
    fingerprints = {}
    _START_CHAR = "<"
    _END_CHAR = ">"

    last_distances = {}

    def __init__(self, path_to_profiles='../profiles/dataset1/val_1/'):
        if not re:
            raise EnvironmentError("classify.textcat requires the regex module that "
                                   "supports unicode. Try '$ pip install regex' and "
                                   "see https://pypi.python.org/pypi/regex for "
                                   "further details.")

        self._corpus = CrubadanCorpusReader(root=path_to_profiles)

        # Load all language ngrams into cache
        for lang in self._corpus.langs():
            self._corpus.lang_freq(lang)

    def remove_punctuation(self, text):
        ''' Get rid of punctuation except apostrophes '''
        return re.sub(r"[^\P{P}\']+", "", text)

    def profile(self, text):
        ''' Create FreqDist of trigrams within text '''
        from nltk import word_tokenize, FreqDist

        clean_text = self.remove_punctuation(text)
        tokens = word_tokenize(clean_text)

        fingerprint = FreqDist()
        for t in tokens:
            token_trigram_tuples = trigrams(self._START_CHAR + t + self._END_CHAR)
            token_trigrams = [''.join(tri) for tri in token_trigram_tuples]

            for cur_trigram in token_trigrams:
                if cur_trigram in fingerprint:
                    fingerprint[cur_trigram] += 1
                else:
                    fingerprint[cur_trigram] = 1

        return fingerprint

    def calc_dist(self, lang, trigram, text_profile):
        """ Calculate the "out-of-place" measure between the
            text and language profile for a single trigram """

        lang_fd = self._corpus.lang_freq(lang)
        dist = 0

        if trigram in lang_fd:
            idx_lang_profile = list(lang_fd.keys()).index(trigram)
            idx_text = list(text_profile.keys()).index(trigram)

            # print(idx_lang_profile, ", ", idx_text)
            dist = abs(idx_lang_profile - idx_text)
        else:
            # Arbitrary but should be larger than
            # any possible trigram file length
            # in terms of total lines
            if PY3:
                dist = maxsize
            else:
                dist = maxint

        return dist

    def lang_dists(self, text):
        ''' Calculate the "out-of-place" measure between
            the text and all languages '''

        distances = {}
        profile = self.profile(text)
        # For all the languages
        for lang in self._corpus._all_lang_freq.keys():
            # Calculate distance metric for every trigram in
            # input text to be identified
            lang_dist = 0
            for trigram in profile:
                lang_dist += self.calc_dist(lang, trigram, profile)

            distances[lang] = lang_dist

        return distances

    def guess_language(self, text):
        ''' Find the language with the min distance
            to the text and return its ISO 639-3 code '''
        self.last_distances = self.lang_dists(text)

        return min(self.last_distances, key=self.last_distances.get)
        #################################################')
