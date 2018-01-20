import csv

import cloudpickle
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics.pairwise import cosine_similarity

# Using the same list of stopwords as the authors of the paper.
# This list doesn't exactly match sklearn's.
STOP_WORDS = [
    "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
    "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
    "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "co",
    "con", "could", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
    "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for",
    "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had",
    "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
    "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest",
    "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
    "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much",
    "must", "my", "myself", "name", "namely", "neither", "nevertheless", "next", "nine", "nobody", "now", "nowhere",
    "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
    "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
    "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
    "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take",
    "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though",
    "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
    "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
    "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
    "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
    "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
]

VOCABULARY_SIZE   = 5000
CSV_HEADLINE_KEY  = 'Headline'
CSV_BODY_ID_KEY   = 'Body ID'
CSV_LABEL_KEY     = 'Stance'
CSV_BODY_TEXT_KEY = 'articleBody'

path_train_stances = './train_stances.csv'
path_train_bodies  = './train_bodies.csv'
path_test_stances  = './test_stances_unlabeled.csv'
path_test_bodies   = './test_bodies.csv'

path_store_train   = './dataset_train_encoded.pyk'
path_store_test    = './dataset_test_encoded.pyk'


# Helper functions to pull data from the disk into memory.
# Caveats: some headers are defined for multiple labels. This code will
# ensure that only one instance for each body and header is kept in memory.


def get_or_put(list_, value):
    """Inserts value in list_ if not preset, returns the index of value."""
    try:
        return list_.index(value)
    except ValueError:
        list_.append(value)
        return len(list_) - 1


def read_stance(filepath, has_labels=True):
    """Reads the stances present on the filepath csv file.

    :return (a,b) where a is a list of dicts that describe each sample and b
    is a list of headlines. The body_id on the elements of a correspond to the
    text on the i-th element of b.
    """
    samples = list()
    headlines = list()

    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        for line in reader:
            headline = line[CSV_HEADLINE_KEY]
            hid = get_or_put(headlines, headline)

            node = {
                'headline': hid,
                'body_id': int(line[CSV_BODY_ID_KEY]),
            }

            if has_labels:
                node['label'] = line[CSV_LABEL_KEY]

            samples.append(node)

    return samples, headlines


def read_bodies(filepath):
    """Produces a dict mapping body ids to the text on the filepath csv file."""
    ordered_bodies = list()
    bodies_index = dict()

    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        for line in reader:
            body_id = int(line[CSV_BODY_ID_KEY])
            body = line[CSV_BODY_TEXT_KEY]

            if body_id not in bodies_index:
                ordered_bodies.append(body)
                bodies_index[body_id] = len(ordered_bodies) - 1

    return ordered_bodies, bodies_index


# Loading training data

def show_statistics(amount_samples, amount_heads, amount_bodies, fold):
    print('Data statistics for {} split:'.format(fold))
    print(' - Amount of samples:   {}'.format(amount_samples))
    print(' - Amount of headlines: {}'.format(amount_heads))
    print(' - Amount of bodies:    {}'.format(amount_bodies))



train_samples, train_headlines = read_stance(path_train_stances)
train_bodies, train_bodies_map = read_bodies(path_train_bodies)

test_samples, test_headlines = read_stance(path_test_stances, False)
test_bodies, test_bodies_map = read_bodies(path_test_bodies)

print()
show_statistics(len(train_samples), len(train_headlines), len(train_bodies), 'train')
print()
show_statistics(len(test_samples), len(test_headlines), len(test_bodies), 'test')

# Fitting vectorizers to build feature vectors for each sample

print('Fitting and transforming TF vectorizer.')
all_train_texts = train_headlines + train_bodies

tf_vectorizer = TfidfVectorizer(max_features=VOCABULARY_SIZE, stop_words=STOP_WORDS, use_idf=False)
all_tfs = tf_vectorizer.fit_transform(all_train_texts)

# WARN: Using test data for training LOL. But I'm just replicating the paper's implementation
print('Fitting and transforming TF-IDF vectorizer.')
all_texts = all_train_texts + test_headlines + test_bodies
tfidf_transformer = TfidfVectorizer(max_features=VOCABULARY_SIZE, stop_words=STOP_WORDS)
_ = tfidf_transformer.fit(all_texts)

# Points to the first headline on the list of TFs computed when fitting the vectorizer.
first_headline = len(train_headlines)

print('Fitting label encoder')
all_labels = list(map(lambda sample: sample['label'], train_samples))
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(all_labels)


def build_vector(tf_head, tf_body, similarity):
    return \
    np.concatenate((tf_head.toarray(), tf_body.toarray(), similarity), axis=1)[
        0]


# Building the feature vectors for train samples
train_feature_samples = list()
for sample in train_samples:
    headline_id = sample['headline']
    body_id = sample['body_id']

    body_index = train_bodies_map[body_id]

    tf_headline = all_tfs[headline_id]
    tf_body = all_tfs[first_headline + body_index]

    # This can be retrieved from indexing as well
    tf_idf_head = tfidf_transformer.transform([train_headlines[headline_id]])
    tf_idf_body = tfidf_transformer.transform([train_bodies[body_index]])

    s = cosine_similarity(tf_idf_head, tf_idf_body, dense_output=True)

    train_feature_samples.append(build_vector(tf_headline, tf_body, s))

# Building feature vectors for test samples

cache_headline_tf = dict()
cache_body_tf = dict()

test_feature_samples = list()
for sample in test_samples:
    headline_id = sample['headline']
    body_id = sample['body_id']

    body_index = test_bodies_map[body_id]

    # getting headline tf/tiidf
    if headline_id not in cache_headline_tf:
        text = [test_headlines[headline_id]]

        tf_headline = tf_vectorizer.transform(text)
        tfidf_headline = tfidf_transformer.transform(text)

        cache_headline_tf[headline_id] = (tf_headline, tfidf_headline)

    tf_headline, tfidf_headline = cache_headline_tf.get(headline_id)

    # getting body tf/tiidf
    if body_index not in cache_body_tf:
        text = [test_bodies[body_index]]

        tf_body = tf_vectorizer.transform(text)
        tfidf_body = tfidf_transformer.transform(text)

        cache_body_tf[body_index] = (tf_body, tfidf_body)

    tf_body, tfidf_body = cache_body_tf.get(body_index)

    # this could be cached as well...
    s = cosine_similarity(tfidf_headline, tfidf_body)

    test_feature_samples.append(build_vector(tf_headline, tf_body, s))


def persist(filepath, X, y=None, encoder=None):
    data = dict(X=X)
    if y is not None:
        data['y'] = y
    if encoder:
        data['label_encoder'] = encoder

    with open(filepath, 'wb') as file:
        cloudpickle.dump(data, file)


# Storing encoded train data on the disk
print('Persisting train data.')
persist(path_store_train, train_feature_samples, encoded_labels, label_encoder)

print('Persisting test data.')
persist(path_store_test, test_feature_samples)

