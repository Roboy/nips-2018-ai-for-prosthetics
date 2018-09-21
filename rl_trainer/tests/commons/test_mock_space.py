from rl_trainer.commons import MockSpace


def test_mock_space_sample_returns_list():
    dim = 3
    sample = MockSpace(dim).sample()
    assert len(sample) == 3
    assert isinstance(sample, list)