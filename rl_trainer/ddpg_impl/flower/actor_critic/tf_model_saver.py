import os

from typeguard import typechecked
import tensorflow as tf


class TFModelSaver:

    @typechecked
    def __init__(self, saving_frequency: int = 1, tf_model_dir: str = "./tf_model_res/"):
        self._saving_frequency = saving_frequency
        self._tf_model_dir = tf_model_dir
        self._saver = tf.train.Saver()
        self._step_counter = 0

    @typechecked
    def step(self, sess: tf.Session):
        self._step_counter += 1
        if self._step_counter % self._saving_frequency == 0:
            self._save(sess)

    def _save(self, sess: tf.Session):
        if not os.path.exists(self._tf_model_dir):
            os.makedirs(self._tf_model_dir)
        self._saver.save(sess=sess, save_path=self._tf_model_dir,
                         global_step=self._step_counter)
