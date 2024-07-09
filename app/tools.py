import pickle

from os.path import join

from app.config import conf


def get_stopwords():
    with open(join(conf.stopwords_file_path, "stopwords.pkl"), "rb") as f:
        return pickle.load(f)


def get_train_text():
    with open(join(conf.datasets_file_path, "train_text.pkl"), "rb") as f:
        return pickle.load(f)


def get_test_text():
    with open(join(conf.datasets_file_path, "test_text.pkl"), "rb") as f:
        return pickle.load(f)
