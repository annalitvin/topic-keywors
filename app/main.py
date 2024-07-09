import joblib

from os.path import join

from app.config import conf
from app.topic_modeler import TopicModeler


if __name__ == "__main__":

    count_vect = joblib.load(join(conf.model_file_path, "countVect.pkl"))
    lda = joblib.load(join(conf.model_file_path, "lda.pkl"))

    modeler = TopicModeler(count_vect, lda)
    keywords = modeler.get_keywords(conf.number_topic, conf.number_keywords)

    joblib.dump(list(keywords), conf.keywords_file_path)
