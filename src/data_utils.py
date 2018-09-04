import os
import numpy as np
import logging
from collections import Counter

import utils

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def create_vocabulary(vocab_path, max_vocab_size, data_paths):
    """Create vocabulary files (if it does not exist yet) from data file."""
    logger = logging.getLogger(__name__)
    if not os.path.exists(vocab_path):
        logger.info("Creating vocabulary %s from data files %s" %
                    (vocab_path, str(data_paths)))
        vocab = {}
        for data_path in data_paths:
            with open(data_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    words = line.split(' ')
                    for word in words:
                        if word in vocab:
                            vocab[word] += 1
                        else:
                            vocab[word] = 1
        logger.info("Finished reading data. %d unique words" % len(vocab))

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if max_vocab_size != -1 and len(vocab_list) > max_vocab_size:
            vocab_list = vocab_list[:max_vocab_size]
        vocab_size = len(vocab_list)
        with open(vocab_path, mode="w") as vocab_file:
            for w in vocab_list:
                if w in _START_VOCAB:
                    vocab_file.write(w + "\t" + "0" + "\n")
                else:
                    vocab_file.write(w + "\t" + str(vocab[w]) + "\n")
    else:
        logger.info('Vocabulary already exists. Skip creating.')
        vocab_size = 0
        with open(vocab_path, "r") as vocab_file:
            for line in vocab_file:
                vocab_size += 1
    logger.info('Finished vocabulary creation.')
    return vocab_size


def create_vocabulary_special(vocab_path, max_vocab_size,
                              data_paths_train, data_paths_test):
    """Create vocabulary files (if it does not exist yet) from data file.

    from training, remove words only occur once and not in word2vec 
    from testing, remove words neither in training nor in word2vec 
    """
    logger = logging.getLogger(__name__)
    if not os.path.exists(vocab_path):
        word2vec = load_word2vec()
        # training and validation
        vocab_train = {}
        for data_path in data_paths_train:
            logger.info("Reading %s" % data_path)
            with open(data_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    words = line.split(' ')
                    for word in words:
                        if word in vocab_train:
                            vocab_train[word] += 1
                        else:
                            vocab_train[word] = 1
        # remove criteria: only occur once in training and not in word2vec
        to_remove = [w for w in vocab_train if vocab_train[w] <= 1]
        to_remove = [w for w in to_remove if w not in word2vec]
        logger.info("%d words in training removed from vocabulary:" %
                    len(to_remove))
        logger.info(utils.format_list(to_remove))
        for w in to_remove:
            del vocab_train[w]

        # testing
        vocab_test = {}
        for data_path in data_paths_test:
            logger.info("Reading %s" % data_path)
            with open(data_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    words = line.split(' ')
                    for word in words:
                        if word in vocab_test:
                            vocab_test[word] += 1
                        else:
                            vocab_test[word] = 1
        # remove criteria: not covered by training and not in word2vec
        to_remove = [w for w in vocab_test
                     if w not in vocab_train and w not in word2vec]
        logger.info("%d words in testing removed from vocabulary:" %
                    len(to_remove))
        logger.info(utils.format_list(to_remove))
        for w in to_remove:
            del vocab_test[w]
        vocab = Counter(vocab_train) + Counter(vocab_test)

        logger.info("Finished reading data. %d unique words "
                    % len(vocab))

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if max_vocab_size != -1 and len(vocab_list) > max_vocab_size:
            vocab_list = vocab_list[:max_vocab_size]
        vocab_size = len(vocab_list)
        with open(vocab_path, mode="w") as vocab_file:
            for w in vocab_list:
                if w in _START_VOCAB:
                    vocab_file.write(w + "\t" + "0" + "\n")
                else:
                    vocab_file.write(w + "\t" + str(vocab[w]) + "\n")
    else:
        logger.info('Vocabulary already exists. Skip creating.')
        vocab_size = 0
        with open(vocab_path, "r") as vocab_file:
            for line in vocab_file:
                vocab_size += 1
    logger.info('Finished vocabulary creation.')
    return vocab_size


def initialize_vocabulary(vocab_path):
    """Initialize vocabulary from file."""
    logger = logging.getLogger(__name__)
    logger.info('Start to initialize vocabulary.')
    if os.path.exists(vocab_path):
        rev_vocab = []
        with open(vocab_path, "r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.split("\t")[0] for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)
    logger.info('Finish vocabulary initialization.')


def sentence_to_token_ids(sentence, vocab):
    """Convert a string to list of integers representing token-ids."""
    return [vocab.get(w, UNK_ID) for w in sentence.strip().split(' ')]


def data_to_token_ids(source_path, target_path, vocab_path,
                      source_ids_path, target_ids_path):
    """Turn data file into token-ids using given vocabulary file."""
    logger = logging.getLogger(__name__)
    if (not os.path.exists(source_ids_path) or
            not os.path.exists(target_ids_path)):
        logger.info("transforming data in %s and %s" %
                    (source_path, target_path))
        vocab, _ = initialize_vocabulary(vocab_path)
        with open(source_path, mode='r') as fread:
            with open(source_ids_path, mode='w') as fwrite:
                for line in fread:
                    ids = sentence_to_token_ids(line, vocab)
                    fwrite.write(utils.format_list(ids) + "\n")
        with open(target_path, mode='r') as fread:
            with open(target_ids_path, mode='w') as fwrite:
                for line in fread:
                    ids = sentence_to_token_ids(line, vocab)
                    fwrite.write(utils.format_list(ids) + "\n")


def prepare_data(data_dir, vocab_file, vocab_size):
    """read data from data_dir, create vocabularies and tokenize data."""
    # Create vocabulary of the appropriate size.
    source_paths = []
    source_paths.append(os.path.join(data_dir, "source.train"))
    source_paths.append(os.path.join(data_dir, "source.valid"))
    source_paths.append(os.path.join(data_dir, "source.test"))
    target_paths = []
    target_paths.append(os.path.join(data_dir, "target.train"))
    target_paths.append(os.path.join(data_dir, "target.valid"))
    target_paths.append(os.path.join(data_dir, "target.test"))
    vocab_path = os.path.join(data_dir, vocab_file)

    # 1) Only use training and validation set to construct vocab
    # vocab_size = create_vocabulary(vocab_path, vocab_size,
    #                                source_paths[:2] + target_paths[:2])

    # 2) Use all data. Note that this is not cheating with using testing data.
    # Words existing only in testing set will not be trained anyway.
    # This saves the labor of generating word2vec embeddings of
    # out-of-vocabulary words in testing time on the fly
    # This is one of the benefits of using pre-trained word embedding
    # that is, to alleviate the vocabulary shifting problem of domain adaptation
    vocab_size = create_vocabulary(vocab_path, vocab_size,
                                   source_paths + target_paths)

    # 3) from training, remove words only occur once and not in word2vec
    # from testing, remove words neither in training nor in word2vec
    # vocab_size = \
    #     create_vocabulary_special(vocab_path, vocab_size,
    #                               source_paths[:2] + target_paths[:2],
    #                               [source_paths[2]] + [target_paths[2]])

    # Create token ids for the data.
    source_ids_paths = []
    source_ids_paths.append(os.path.join(data_dir, "source.train.id"))
    source_ids_paths.append(os.path.join(data_dir, "source.valid.id"))
    source_ids_paths.append(os.path.join(data_dir, "source.test.id"))
    target_ids_paths = []
    target_ids_paths.append(os.path.join(data_dir, "target.train.id"))
    target_ids_paths.append(os.path.join(data_dir, "target.valid.id"))
    target_ids_paths.append(os.path.join(data_dir, "target.test.id"))
    for i in range(3):
        data_to_token_ids(source_paths[i], target_paths[i], vocab_path,
                          source_ids_paths[i], target_ids_paths[i])

    return source_ids_paths, target_ids_paths, vocab_size


def read_embeddings(lexical_embedding_path):
    """read word embeddings"""
    embeddings = np.load(lexical_embedding_path)
    return embeddings


def load_word2vec():
    '''load pre-trained word2vec embeddings'''
    from gensim.models import word2vec
    word2vec_path = '../../shared/word2vec/GoogleNews-vectors-negative300.bin'
    model = word2vec.Word2Vec.load_word2vec_format(word2vec_path, binary=True)
    return model
