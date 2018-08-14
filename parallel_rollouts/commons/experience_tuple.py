from typing import Sequence


class ExperienceTuple:

    DIM_STATE = 158
    DIM_ACTION = 19

    def __init__(
            self,
            initial_state: Sequence[float],
            action: Sequence[float],
            final_state: Sequence[float],
    ):
        assert len(initial_state) == self.DIM_STATE
        assert len(action) == self.DIM_ACTION
        assert len(final_state) == self.DIM_STATE

        self.initial_state = initial_state
        self.action = action
        self.final_state = final_state
