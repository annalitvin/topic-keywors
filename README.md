# Topic Keywords

#### This project provides tools to extract keywords on a specific topic.

## Run App
### Local

```sh
$ sh run.sh
```

```sh
$ poetry run python -m app.preprocessing
$ poetry run python -m app.training_pipeline
$ poetry run python -m app.main
```


## Topic keywords extraction workflow

1. Need to prepare the dataset (reprocessing): removed stop words, tokenization and lemmatization, deleted punctuation marks and small text.
2. Next, we train the model using a custom dictionary and save the results in **lda.pkl**, **countVect.pkl**, **tf_idf.pkl** (training_pipeline).
3. At the end, we test the model on a test dataset and return the top keywords for each topic.
The result is serialized in **keywords.pkl** file.

**Output example :**
```
Topic: 1:  ['gpu', 'cpu', 'gpus', 'time', 'aws', 'power', 'running', 'card', 'performance', 'box']
Topic: 2:  ['image', 'feature', 'output', 'input', 'code', 'layer', 'dataset', 'time', 'number', 'size']
Topic: 3:  ['man', 'game', 'video', 'code', 'time', 'software', 'layer', 'advantage', 'machine', 'global']
Topic: 4:  ['text', 'time', 'word', 'machine', 'music', 'face', 'sentence', 'translation', 'generate', 'set']
Topic: 5:  ['cnn', 'function', 'image', 'output', 'layer', 'input', 'gradient', 'neuron', 'step', 'object']
Topic: 6:  ['tensorflow', 'app', 'action', 'time', 'table', 'environment', 'architecture', 'state', 'experience', 'trained']
Topic: 7:  ['machine', 'memory', 'human', 'language', 'search', 'time', 'attention', 'voice', 'field', 'translation']
Topic: 8:  ['game', 'policy', 'function', 'play', 'state', 'probability', 'alphago', 'cluster', 'community', 'time']
Topic: 9:  ['human', 'time', 'technology', 'computer', 'software', 'process', 'user', 'machine', 'story', 'facebook']
Topic: 10:  ['machine', 'python', 'free', 'average', 'university', 'weighted', 'programming', 'week', 'github', 'lot']

```
**File output:**
```
 keywords.pkl
```

## Development
### Run Tests and Linter

```
$ poetry run tox
```
