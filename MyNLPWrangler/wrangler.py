from urllib.parse import urlparse
import re
import string
import pandas as pd
import numpy as np


class ArticleCleaner():
    """
    clean data  for nlp
    """

    def __init__(self, df: pd.DataFrame, col: str):
        self._df = df
        self._col = col

    def is_url(self, url):
        """
        remove the url from the post
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def remove_punctuation(self, line):
        # remove punctuation from the post
        rule = re.compile("[^\u4e00-\u9fa5^.^a-z^A-Z]")
        line = rule.sub(' ', line)
        line = re.sub('[%s]' % re.escape(string.punctuation), '', line)
        return line

    def clean_data(self):
        """
        for nlp clean data,it included remove url and puntuation
        """
        next_col = self._col + '_new'
        self._df[self._col] = self._df[self._col].replace('\r', '', regex=True)
        self._df = self._df.dropna(subset=[self._col])
        self._df[next_col] = [
            ' '.join(y for y in x.split() if not self.is_url(y)) for x in self._df[self._col]]
        self._df[next_col] = self._df[next_col].replace('\n', ' ', regex=True)
        self._df[next_col] = self._df[next_col].apply(self.remove_punctuation)
        self._df = self._df.replace(r'^\s*$', np.nan, regex=True)
        self._df = self._df.dropna(subset=[next_col])
        self._df.drop_duplicates(subset=[next_col], keep='last', inplace=True)
        return self._df
