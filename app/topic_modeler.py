import joblib

from typing import Iterable
from os.path import join

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

from app.config import conf


class TopicModeler:
    """
    Inteface object for CountVectorizer + LDA simple usage.
    """

    def __init__(self, count_vect: CountVectorizer, lda: LDA):
        """
        Args:
             count_vect - CountVectorizer object from sklearn.
             lda - LDA object from sklearn.
        """
        self.lda = lda
        self.count_vect = count_vect
        self.count_vect.input = "content"

    def get_keywords(self, n_topics: int, n_keywords: int) -> Iterable:
        """
        For a given text gives n top keywords for each of m top texts topics.
        Args:
             n_topics: int - how many top topics to use.
             n_keywords: int - how many top words of each topic to return.
        returns:
            Iterable - generator with n top keywords for each of m top texts topics
            E.g. {'topic': 6, 'top_terms': ['machine', 'python', 'free', 'average', 'weighted',
            'programming', 'university', 'week', 'scikit', 'supervised']}
        """
        terms = self.count_vect.get_feature_names_out()
        lda_components = self.lda.components_[:n_topics]

        for index, component in enumerate(lda_components, 1):
            zipped = zip(terms, component)
            top_terms_key = sorted(zipped, key=lambda t: t[1], reverse=True)[:n_keywords]
            top_terms_list = list(dict(top_terms_key).keys())
            yield dict(topic=index, top_terms=top_terms_list)


if __name__ == "__main__":

    count_vect = joblib.load(join(conf.model_file_path, "countVect.pkl"))
    lda = joblib.load(join(conf.model_file_path, "lda.pkl"))

    modeler = TopicModeler(count_vect, lda)
    keywords = modeler.get_keywords(n_topics=conf.number_topic, n_keywords=conf.number_keywords)
    for topic_item in keywords:
        print(f"Topic: {topic_item['topic']}: ", topic_item["top_terms"])
