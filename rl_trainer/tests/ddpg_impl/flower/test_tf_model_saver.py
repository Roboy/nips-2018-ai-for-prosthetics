import os
import shutil

import pytest

from rl_trainer.ddpg_impl.flower.actor_critic.tf_model_saver import TFModelSaver
import tensorflow as tf

MODEL_DIR = "./delme_tf_model/"


@pytest.fixture(scope="module")
def tf_session():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    return sess


def setup():
    assert not os.path.exists(MODEL_DIR), "Target saving folder already exists"


def cleanup():
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)


def test_construction_doesnt_create_dir():
    setup()
    TFModelSaver(tf_model_dir=MODEL_DIR)
    assert not os.path.exists(MODEL_DIR)


def test_lazy_dir_creation(tf_session):
    setup()
    try:
        saver = TFModelSaver(saving_frequency=1, tf_model_dir=MODEL_DIR)
        saver.step(sess=tf_session)
        assert os.path.exists(MODEL_DIR)
    finally:
        cleanup()


def test_saving_frequency(tf_session):
    setup()
    saving_frequency = 3
    try:
        saver = TFModelSaver(saving_frequency=saving_frequency, tf_model_dir=MODEL_DIR)
        for _ in range(saving_frequency - 1):
            saver.step(sess=tf_session)
            assert not os.path.exists(MODEL_DIR), f"{MODEL_DIR} exists"
        saver.step(sess=tf_session)
        assert os.path.exists(MODEL_DIR)
    finally:
        cleanup()
