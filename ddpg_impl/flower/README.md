Based on the implementation in: https://github.com/pemami4911/deep-rl
# Launch flower
From the ``ddpg_impl/`` directory launch:
```bash
python -m flower.main
```
To get visualization and dump them to ``.mp4`` files:
```bash
python -m flower.main --use-gym-monitor --render-env
```
See further command line options in ``flower/arg_parser.py``

# Testing

```bash
pytest
```