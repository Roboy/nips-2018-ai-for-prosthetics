import csv
from typing import Sequence, List

from rl_trainer.commons import Episode


class CSVEpisodeSerializer:

    def serialize(self, episode: Episode, out_fname: str):
        with open(out_fname + ".csv", "w", newline='') as file:
            writer = csv.writer(file)
            for exp_tuple in episode.experience_tuples:
                line = [
                    self._format_line(exp_tuple.initial_state),
                    self._format_line(exp_tuple.action),
                    self._format_line(exp_tuple.final_state),
                    self._format_line([exp_tuple.reward]),
                ]
                writer.writerow(line)

    @staticmethod
    def _format_line(line: Sequence[float]) -> List[str]:
        return [f"{float(e):.4}" for e in line]
