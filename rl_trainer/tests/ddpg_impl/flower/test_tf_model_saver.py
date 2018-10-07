import os
import shutil

import pytest

from rl_trainer.ddpg_impl.flower.actor_critic.tf_model_saver import TFModelSaver
import tensorflow as tf

MODEL_DIR = "./delme_tf_model/"


@pytest.fixture(scope="module")
def tf_session():
    sess = tf.Session(graph=tf.Graph())
    with sess.graph.as_default():
        tf.Variable(1)
        sess.run(tf.global_variables_initializer())
    return sess


def model_dir_doesnt_exist():
    return not os.path.exists(MODEL_DIR), "Target saving folder already exists"


def remove_model_dir():
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)


def test_construction_doesnt_create_dir():
    assert model_dir_doesnt_exist()
    TFModelSaver(tf_model_dir=MODEL_DIR)
    assert not os.path.exists(MODEL_DIR)


def test_lazy_dir_creation(tf_session: tf.Session):
    assert model_dir_doesnt_exist()
    try:
        with tf_session.graph.as_default():
            saver = TFModelSaver(saving_frequency=1, tf_model_dir=MODEL_DIR)
        saver.step(sess=tf_session)
        assert os.path.exists(MODEL_DIR)
    finally:
        remove_model_dir()


def test_saving_frequency(tf_session: tf.Session):
    assert model_dir_doesnt_exist()
    saving_frequency = 3
    try:
        with tf_session.graph.as_default():
            saver = TFModelSaver(saving_frequency=saving_frequency, tf_model_dir=MODEL_DIR)
        for _ in range(saving_frequency - 1):
            saver.step(sess=tf_session)
            assert model_dir_doesnt_exist()
        saver.step(sess=tf_session)
        assert os.path.exists(MODEL_DIR)
    finally:
        remove_model_dir()
