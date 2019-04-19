# This file has the code to train the LDA model using gensim.

import pandas as pd
import gensim
from gensim.test.utils import datapath
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()

def get_tweets():
    """
    Returns English language tweets.
    """
    # Load the DataFrame:
    df = pd.read_csv('./20190406_AM_Data.csv')
    # Filter by english language:
    df = df.loc[df['tweetLang'].str.startswith('en')]
    # Get tweets:
    tweets = df['tweetText'].tolist()
    return tweets

def prepare_stopwords():
    """
    Returns a list of stopwords (from nltk + some custom stopwords)
    """
    # Prepare stopwords list from four languages + some custom words:
    languages = ['english', 'french', 'german', 'spanish']
    STOPWORDS = []
    for lang in languages:
        STOPWORDS.extend(stopwords.words(lang))
    custom_stopwords = ["amp", "n't", "rt", "http", "https", "migration", "migrant"]
    STOPWORDS.extend(custom_stopwords)
    STOPWORDS = set(STOPWORDS)
    print("4 languages + custom stopwords, we have a total of {} stopwords.".format(len(STOPWORDS)))
    return STOPWORDS


def pre_process_text(tweet, STOPWORDS):
    """
    Takes in a tweet and the list of stopwords as input. Returns the processed tweet as a list of words.
    Processing steps include: word tokenization, lowercase, stopword removal, lemmatization.
    """
    output_words = []

    # Tokenize:
    words = word_tokenize(unicode(tweet, errors='ignore'))

    for word in words:
        word = word.lower() # lowercase
        if word[0].isalpha() and word not in STOPWORDS: # stopword removal
            w = lemmatizer.lemmatize(word) # lemmatization
            output_words.append(w)

    return output_words



if __name__ == '__main__':
    tweets = get_tweets()
    STOPWORDS = prepare_stopwords()

    processed_tweets = []
    for tweet in tweets:
        processed_tweets.append(pre_process_text(tweet, STOPWORDS))
    print("Number of tweets: ", len(processed_tweets))

    # Next steps are inspired from: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
    # Create Dictionary
    id2word = gensim.corpora.Dictionary(processed_tweets)
    # Create Corpus
    texts = processed_tweets
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=5,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=20,
                                               alpha='auto',
                                               per_word_topics=True)

    # Save model to disk.
    temp_file = datapath("5_topic_LDA")
    lda_model.save(temp_file)
