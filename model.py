import numpy as np
import os
import tensorflow as tf
from data_utils import minibatches, pad_sequences, get_chunks
from general_utils import Progbar, print_sentence


class PhoneModel(object):
    def __init__(self, config, nphones, logger=None):
        """
        Args:
            config: class with hyper parameters
            nphones: the number of phones
            logger: logger instance
        """
        self.config     = config

        if logger is None:
            logger = logging.getLogger('logger')
            logger.setLevel(logging.DEBUG)
            logging.basicConfig(format='%(message)s', level=logging.DEBUG)

        self.logger = logger
        self.nphones = nphones


    def add_placeholders(self):
        """
        Adds placeholders to self
        """

        # shape = (batch size, max length of sequence in batch, features)
        self.frames = tf.placeholder(tf.float32, shape=[None, None, 123],
                        name="feats")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        # shape = (batch size, max length of sequence in batch)
        self.phones = tf.placeholder(tf.int32, shape=[None, None],
                        name="phones")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")


    def get_feed_dict(self, framelist, phones=None, lr=None, dropout=None):
        """
        Given some data, pad it and build a feed dictionary
        Args:
            framelist: list of frames. A frame is a list of 123 features.
            phones: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob
        Returns:
            dict {placeholder: value}
        """
        frames, sequence_lengths = pad_sequences(framelist, 0)

        # build feed dictionary
        feed = {
            self.frames: frames,
            self.sequence_lengths: sequence_lengths
        }

        if phones is not None:
            phones, _ = pad_sequences(phones, 0)
            feed[self.phones] = phones

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def add_logits_op(self):
        """
        Adds logits to self
        """
        with tf.variable_scope("bi-lstm"):
            lstm_cell_fwd = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
            lstm_cell_bwd = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fwd,
                lstm_cell_bwd, self.frames, sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.config.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", shape=[2*self.config.hidden_size, self.nphones],
                dtype=tf.float32)

            b = tf.get_variable("b", shape=[self.nphones], dtype=tf.float32,
                initializer=tf.zeros_initializer())

            ntime_steps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, ntime_steps, self.nphones])

    def add_pred_op(self):
        """
        Adds phones_pred to self
        """
        self.phones_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

    def add_loss_op(self):
        """
        Adds loss to self
        """
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.phones)
        mask = tf.sequence_mask(self.sequence_lengths)
        losses = tf.boolean_mask(losses, mask)
        self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


    def add_train_op(self):
        """
        Add train_op to self
        """
        with tf.variable_scope("train_step"):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)


    def add_init_op(self):
        self.init = tf.global_variables_initializer()


    def add_summary(self, sess):
        # tensorboard stuff
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path, sess.graph)


    def build(self):
        self.add_placeholders()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()
        self.add_init_op()


    def predict_batch(self, sess, framelist):
        """
        Args:
            sess: a tensorflow session
            framelist: list of frames
        Returns:
            phones_pred: list of phones for each sentence
            sequence_length
        """
        fd, sequence_lengths = self.get_feed_dict(framelist, dropout=1.0)

        phones_pred = sess.run(self.phones_pred, feed_dict=fd)

        return phones_pred, sequence_lengths


    def run_epoch(self, sess, train, dev, epoch):
        """
        Performs one complete pass over the train set and evaluate on dev
        Args:
            sess: tensorflow session
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) number of the epoch
        """
        nbatches = (len(train) + self.config.batch_size - 1) / self.config.batch_size
        prog = Progbar(target=nbatches)
        for i, (framelist, phones) in enumerate(minibatches(train, self.config.batch_size)):
            fd, _ = self.get_feed_dict(framelist, phones, self.config.lr, self.config.dropout)

            _, train_loss, summary = sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        acc, f1 = self.run_evaluate(sess, dev)
        self.logger.info("- dev acc {:04.2f} - f1 {:04.2f}".format(100*acc, 100*f1))
        return acc, f1


    def run_evaluate(self, sess, test):
        """
        Evaluates performance on test set
        Args:
            sess: tensorflow session
            test: dataset that yields tuple of sentences, tags
        Returns:
            accuracy
            f1 score
        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for framelist, phones in minibatches(test, self.config.batch_size):
            phones_pred, sequence_lengths = self.predict_batch(sess, framelist)

            for lab, lab_pred, length in zip(phones, phones_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += map(lambda a, b: a == b, zip(lab, lab_pred))

                lab_chunks = set(get_chunks(lab, tags))
                lab_pred_chunks = set(get_chunks(lab_pred, tags))
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        return acc, f1


    def train(self, train, dev):
        """
        Performs training with early stopping and lr exponential decay
        Args:
            train: dataset that yields tuple of frame, phoneindex
            dev: dataset
        """
        best_score = 0
        saver = tf.train.Saver()
        # for early stopping
        nepoch_no_imprv = 0
        with tf.Session() as sess:
            sess.run(self.init)
            # tensorboard
            self.add_summary(sess)
            for epoch in range(self.config.nepochs):
                self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.nepochs))

                acc, f1 = self.run_epoch(sess, train, dev, epoch)

                # decay learning rate
                self.config.lr *= self.config.lr_decay

                # early stopping and saving best parameters
                if f1 >= best_score:
                    nepoch_no_imprv = 0
                    if not os.path.exists(self.config.model_output):
                        os.makedirs(self.config.model_output)
                    saver.save(sess, self.config.model_output)
                    best_score = f1
                    self.logger.info("- new best score!")

                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                        self.logger.info("- early stopping {} epochs without improvement".format(
                                        nepoch_no_imprv))
                        break



