import nltk
import pandas as pd
import pickle

from os.path import join

from nltk.corpus import stopwords as nltk_stopwords
from sklearn.model_selection import train_test_split

from app.config import conf


def serialize(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def upload_documents(file_path):
    """Upload text of articles from dataset"""
    data = pd.read_csv(file_path, encoding="utf8")
    documents_list = data["text"].tolist()
    return documents_list


def upload_stopwords():
    """Load stopwords from all the sources we got"""
    extra_stopwords = set()
    stopwords_file = join(conf.raw_stopwords_file_path, "stopwords.txt")
    with open(stopwords_file, "r") as f:
        words = f.read()
        stop_words = [word.strip() for word in words.split(",")]
        extra_stopwords.update(set(stop_words))

    stopwords = set()
    stopwords.update(set(nltk_stopwords.words("english")))
    stopwords.update(extra_stopwords)
    stopwords.update(conf.custom_stopwords)
    stopwords_list = list(stopwords)

    stopwords_dump_file = join(conf.stopwords_file_path, "stopwords.pkl")
    serialize(stopwords_list, stopwords_dump_file)
    return stopwords


def delete_non_letters(words):
    new_words = []
    for word in words:
        new_word = "".join(c for c in word if c.isalpha())
        if new_word:
            new_words.append(new_word)
    return new_words


def normalize(text, tokenized=False, del_stopwords=False):
    if not tokenized:
        text = nltk.word_tokenize(text)
    text = delete_non_letters(text)
    text = [word for word in text if len(word) > 1]
    return text


def split_dataset(dataset):
    """Divide the dataset into training and test"""
    train_text, test_text = train_test_split(dataset, test_size=0.1, random_state=666)

    serialize(train_text, join(conf.datasets_file_path, "train_text.pkl"))
    serialize(test_text, join(conf.datasets_file_path, "test_text.pkl"))
    return train_text, test_text
