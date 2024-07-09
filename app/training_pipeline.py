import re
import numpy as np
import joblib

from os.path import join
from typing import List, Text

from nltk import WordNetLemmatizer, RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

from app.config import conf
from app.tools import get_stopwords
from app.tools import get_train_text


class TextPreparation:

    def __init__(
        self,
        text: List[Text],
        stop_words: List[Text],
        tokenizer: RegexpTokenizer = RegexpTokenizer(r"\w+"),
        lemmatizer: WordNetLemmatizer = WordNetLemmatizer(),
        lower_thresh: float = 2.0,
        upper_thresh: float = 4.0,
    ):
        self.text: List[Text] = text
        self.stop_words: List[Text] = stop_words
        self.tokenizer: RegexpTokenizer = tokenizer
        self.lemmatizer: WordNetLemmatizer = lemmatizer
        self.lower_thresh: float = lower_thresh
        self.upper_thresh: float = upper_thresh
        self.tf_idf: TfidfVectorizer = self.__get_tf_idf()
        self.__words: List[str] = self.tf_idf.get_feature_names_out()

    @property
    def words(self) -> List[str]:
        return self.get_good_words(self.__words)

    def __get_tf_idf(self) -> TfidfVectorizer:
        """
        Convert a collection of raw documents to a matrix of TF-IDF features
        returns:
             TfidfVectorizer - fitted vectorizer
        """
        tf_idf = TfidfVectorizer(stop_words=self.stop_words, smooth_idf=False, tokenizer=self.tokenizer.tokenize)
        tf_idf.fit(self.text)
        return tf_idf

    def get_good_words(self, words) -> List[str]:
        """
        Sorting out too rare and too common words and lemmatization.
        Args:
            words - raw words that are obtained from the dataset.
        returns:
            list - list of cleaned words.
        """
        tf_idfs = self.tf_idf.idf_
        not_often = tf_idfs > self.lower_thresh
        not_rare = tf_idfs < self.upper_thresh
        mask = not_often * not_rare
        return self.lemmatize(np.array(words)[mask])

    @staticmethod
    def __delete_punctuation(word) -> str:
        """
        Delete punctuation from word.
        Args:
            word - raw word that are obtained from the dataset.
        returns:
            str - word without punctuation.
        """
        return re.sub(r"^(\d+\w*$|_+)", "", word)

    def lemmatize(self, words) -> List[str]:
        """
        To reduce the different forms of a word to one single form,
        for example, reducing "builds", "building",or "built" to the lemma "build":
        Compounds were lemmatized, that is, inflectional differences were disregarded.
        Args:
            words: numpy array - words that are obtained from the dataset.
        returns:
            list - lemmatized words.
        """
        lemmatized_words = set()
        for word in words:
            word = self.__delete_punctuation(word)
            lemmatized_word = self.lemmatizer.lemmatize(word)
            if len(lemmatized_word) <= 2:
                continue
            lemmatized_words.add(lemmatized_word)
        return list(lemmatized_words)


if __name__ == "__main__":
    stopwords = get_stopwords()
    train_text = get_train_text()

    text_preparation = TextPreparation(text=train_text, stop_words=stopwords)
    vocabulary = {word: i for i, word in enumerate(text_preparation.words)}

    count_vect = CountVectorizer(stop_words=stopwords, vocabulary=vocabulary)
    dataset = count_vect.fit_transform(train_text)
    lda = LDA(n_components=conf.number_topic, max_iter=30, n_jobs=6, learning_method="batch", verbose=1)
    lda.fit_transform(dataset)

    joblib.dump(lda, join(conf.model_file_path, "lda.pkl"))
    joblib.dump(count_vect, join(conf.model_file_path, "countVect.pkl"))
    joblib.dump(text_preparation.tf_idf, join(conf.model_file_path, "tf_idf.pkl"))
