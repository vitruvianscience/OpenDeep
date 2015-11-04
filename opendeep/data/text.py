"""
Generic structure for a dataset reading from a file or directory.
"""
# standard libraries
import logging
import time
import warnings
import itertools
# third party
import numpy
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
# internal imports
from opendeep.data.dataset_file import FileDataset
from opendeep.data.stream.filestream import FileStream
from opendeep.data.stream.modifystream import ModifyStream
from opendeep.data.stream.batchstream import BufferStream
from opendeep.utils.misc import numpy_one_hot, make_time_units_string, compose

log = logging.getLogger(__name__)

class TextDataset(FileDataset):
    """
    This gives a file-based dataset for working with text (either characters or words).
    It will construct a vocabulary dictionary with each token.
    """
    def __init__(self, path, source=None, train_filter=None, valid_filter=None, test_filter=None,
                 inputs_preprocess=None, targets_preprocess=None,
                 vocab=None, label_vocab=None, unk_token="<UNK>", level="char", target_n_future=None,
                 sequence_length=False):
        """
        Initialize a text-based dataset. It will output one-hot vector encodings for the appropriate level (word,
        char, line).

        Parameters
        ----------
        path : str
            The name of the file or directory for the dataset.
        source : str, optional
            The URL path for downloading the dataset (if applicable).
        train_filter : regex string or compiled regex object, optional
            The regular expression filter to match training file names against (if applicable).
        valid_filter : regex string or compiled regex object, optional
            The regular expression filter to match validation file names against (if applicable).
        test_filter : regex string or compiled regex object, optional
            The regular expression filter to match testing file names against (if applicable).
        inputs_preprocess : function, optional
            A preprocessing function to apply to input data. This function will be applied to each line
            from the files in `path`, and if it creates a list of elements, each element will be yielded as the
            input data separately. For example, the function could be ``lambda line: (line.split(',')[0]).lower()``
            to grab a string before a comma on each line and lowercase it. Preprocessing will happen before any
            tokenization is applied i.e. tokenizing and processing are composed as tokenize(preprocess(line)).
        targets_preprocess : function, optional
            A preprocessing function to apply to targets data. This function will be applied to each line from
            the files in `path`, and if it creates a list of elements, each element will be yielded as the target
            label data separately. For example, the function could be ``lambda line: (line.split(',')[1]).lower()``
            to grab a label after a comma on each line and lowercase it. Tokenization will not be applied to data
            yielded from the targets' preprocessing.
        vocab : dict, optional
            A starting dictionary to use when converting tokens to numbers.
        label_vocab : dict, optional
            A starting dictionary to use when converting labels (targets) to numbers.
        unk_token : str
            The representation for an unknown token to use in the vocab dictionary.
        level : str
            Either ``char``, ``word``, or ``line``, saying how to process the text.
            For ``char``, data will be character-level.
            For ``word``, data will be split by whitespace.
            For ``line``, data will be split by newline.
        target_n_future : int, optional
            For creating language models that predict tokens in the future, this determines the skip size (number of
            steps in the future) that the language model will try to predict as its target. Most language models will
            have target_n_future=1. If `target_n_future` is not None, the targets will be created from the inputs
            (but still apply targets_preprocess instead of inputs_preprocess if it is different).
        sequence_length : int, optional
            The maximum length of subsequences to iterate over this dataset. If this is None or False, the data
            will just be supplied as a stream of one-hot vectors rather than broken into 2-D one-hot vector sequences.
        """
        # Figure out if we want characters, words, or lines processed, and create the processing function
        # to compose on top of the preprocessing function arguments.
        level = level.lower()
        if level == "char":
            tokenize = lambda s: list(s)
        elif level == "word":
            if NLTK_AVAILABLE:
                tokenize = lambda s: nltk.tokenize.word_tokenize(s)
            else:
                warnings.warn("NLTK isn't installed - going to split strings by whitespace. Highly recommended "
                              "that you install nltk for better word tokenization.")
                tokenize = lambda s: s.split()
        elif level == "line":
            tokenize = lambda s: [s]
        else:
            tokenize = None

        if sequence_length:
            assert sequence_length > 1, "Need to have a sequence_length greater than 1, found %d" % sequence_length
        self.sequence_len = sequence_length

        # modify our file stream's processors to work with the appropriate level!
        # if target_n_future is not none, we are assuming that this is a language model and that we
        # should tokenize the target
        if target_n_future is not None:
            targets_preprocess = compose(tokenize, inputs_preprocess)
        inputs_preprocess = compose(tokenize, inputs_preprocess)

        # call super to create the data streams
        super(TextDataset, self).__init__(path=path, source=source,
                                          train_filter=train_filter, valid_filter=valid_filter, test_filter=test_filter,
                                          inputs_preprocess=inputs_preprocess, targets_preprocess=targets_preprocess)
        # after this call, train_inputs, train_targets, etc. are all lists or None.

        # determine if this is a language model, and adjust the stream accordingly to use the inputs as the targets
        if target_n_future is not None:
            self.train_targets = FileStream(path, train_filter, targets_preprocess, target_n_future)
            if valid_filter is not None:
                self.valid_targets = FileStream(path, valid_filter, targets_preprocess, target_n_future)
            if test_filter is not None:
                self.test_targets = FileStream(path, test_filter, targets_preprocess, target_n_future)

        # Create our vocab dictionary if it doesn't exist!
        self.unk_token = unk_token
        vocab_inputs = [self.train_inputs] + (self.valid_inputs or [])
        self.vocab = vocab or self.compile_vocab(itertools.chain(*vocab_inputs))
        vocab_len = len(self.vocab)
        self.vocab_inverse = {v: k for k, v in self.vocab.items()}

        # Now modify our various inputs streams with one-hot versions using the vocab dictionary.
        # (making sure they remain as lists to satisfy the superclass condition)
        rep = lambda token: self.vocab.get(token, self.vocab.get(self.unk_token))
        one_hot = lambda token: numpy_one_hot([rep(token)], n_classes=vocab_len)[0]
        self.train_inputs = ModifyStream(self.train_inputs, one_hot)
        if self.sequence_len:
            self.train_inputs = self._subsequence(self.train_inputs)
        if self.valid_inputs is not None:
            self.valid_inputs = ModifyStream(self.valid_inputs, one_hot)
            if self.sequence_len:
                self.valid_inputs = self._subsequence(self.valid_inputs)
        if self.test_inputs is not None:
            self.test_inputs = ModifyStream(self.test_inputs, one_hot)
            if self.sequence_len:
                self.valid_inputs = self._subsequence(self.valid_inputs)

        # Now deal with possible output streams (either tokenizing it using the supplied label dictionary,
        # creating the label dictionary, or using the vocab dictionary if it is a language model
        # (target_n_future is not none)
        if self.train_targets is not None and target_n_future is None:
            vocab_inputs = [self.train_targets] + (self.valid_targets or [])
            self.label_vocab = label_vocab or \
                               self.compile_vocab(itertools.chain(*vocab_inputs))
            self.label_vocab_inverse = {v: k for k, v in self.label_vocab.items()}
        # if this is a language model, label vocab is same as input vocab
        elif target_n_future is not None:
            self.label_vocab = self.vocab
            self.label_vocab_inverse = self.vocab_inverse
        else:
            self.label_vocab = None
            self.label_vocab_inverse = None

        # now modify the output streams with the one-hot representation using the vocab (making sure they remain
        # as lists to satisfy the superclass condition)
        if self.label_vocab is not None:
            label_vocab_len = len(self.label_vocab)
            label_rep = lambda token: self.label_vocab.get(token, self.label_vocab.get(self.unk_token))
            label_one_hot = lambda token: numpy_one_hot([label_rep(token)], n_classes=label_vocab_len)[0]
            if self.train_targets is not None:
                self.train_targets = ModifyStream(self.train_targets, label_one_hot)
                if self.sequence_len:
                    self.train_targets = self._subsequence(self.train_targets)
            if self.valid_targets is not None:
                self.valid_targets = ModifyStream(self.valid_targets, label_one_hot)
                if self.sequence_len:
                    self.valid_targets = self._subsequence(self.valid_targets)
            if self.test_targets is not None:
                self.test_targets = ModifyStream(self.test_targets, label_one_hot)
                if self.sequence_len:
                    self.test_targets = self._subsequence(self.test_targets)

    def _subsequence(self, stream):
        numpy_concat = lambda l: numpy.vstack(l)
        return ModifyStream(
            BufferStream(stream, self.sequence_len),
            numpy_concat
        )

    def compile_vocab(self, iters):
        """
        Creates a dictionary mapping tokens (words or characters) to integers given the level and preprocessing.

        Parameters
        ----------
        iters : iterable
            The iterable to go through when creating the vocaublary dictionary.

        Returns
        -------
        vocab
            The dictionary mapping token: integer for all tokens in the `iters` iterable.
        """
        log.debug("Creating vocabulary...")
        t = time.time()
        vocab = {self.unk_token: 0}
        i = 1
        for token in iters:
            if token not in vocab:
                vocab[token] = i
                i += 1
        log.debug("Vocab took %s to create." % make_time_units_string(time.time() - t))
        return vocab
