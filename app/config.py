from os import path
from typing import Set

from pydantic import BaseConfig


PARENT_DIR = path.dirname(path.abspath(__file__))


class TopicKeywordsConfig(BaseConfig):
    number_topic = 10
    number_keywords = 10

    stopwords_file_path: str = path.join(PARENT_DIR, "stopwords")
    raw_stopwords_file_path: str = path.join(stopwords_file_path, "raw_stopwords")

    custom_stopwords: Set[str] = {"http", "https", "com"}

    datasets_file_path: str = path.join(PARENT_DIR, "datasets")
    raw_dataset_file_path: str = path.join(datasets_file_path, "raw_dataset")
    dataset_filename: str = "articles.csv"

    model_file_path: str = path.join(PARENT_DIR, "model")

    keywords_file_path: str = path.join(PARENT_DIR, "keywords", "keywords.pkl")


conf = TopicKeywordsConfig()
