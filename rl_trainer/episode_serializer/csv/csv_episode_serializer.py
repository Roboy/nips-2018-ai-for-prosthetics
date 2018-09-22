import csv
from typing import List, Collection

from rl_trainer.commons import Episode


class CSVEpisodeSerializer:

    def serialize(self, episode: Episode, out_fname: str):
        with open(out_fname + ".csv", "w", newline='') as file:
            writer = csv.writer(file)
            for exp_tuple in episode.experience_tuples:
                line = [
                    self._format_nums(exp_tuple.state_1),
                    self._format_nums(exp_tuple.action),
                    f"{float(exp_tuple.reward):.6}",
                    self._format_nums(exp_tuple.state_2),
                    str(exp_tuple.state_2_is_terminal),
                ]
                writer.writerow(line)

    @staticmethod
    def _format_nums(nums: Collection[float]) -> List[str]:
        return [f"{float(e):.6}" for e in nums]
