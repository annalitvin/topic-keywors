import nltk
import numpy as np

from os.path import join

from app.config import conf

from app.preprocessing_tools import upload_documents
from app.preprocessing_tools import upload_stopwords
from app.preprocessing_tools import normalize
from app.preprocessing_tools import split_dataset


def create_training_data(dataset_file_path):
    """
    Create training data to train LDA model.

    Data preparation for model training includes the following steps:
    1. loading documents from the dataset
    2. loading stopwords from all the sources we got
    3. deleted small text that is less than 100 words long
    4. divided the dataset into training and test

    Args:
         dataset_file_path - file path to the dataset location
    returns:
        None
    """
    documents_list = upload_documents(dataset_file_path)
    upload_stopwords()

    lower_text_size = 100
    texts_lens = []
    for document in documents_list:
        tok_text = nltk.word_tokenize(document)
        tok_text = normalize(tok_text, tokenized=True)
        texts_lens.append(len(tok_text))

    long_texts_mask = np.array(texts_lens) > lower_text_size
    documents_list = np.array(documents_list)[long_texts_mask]
    split_dataset(documents_list)


if __name__ == "__main__":
    articles_dataset_file = join(conf.raw_dataset_file_path, conf.dataset_filename)
    create_training_data(articles_dataset_file)
