import csv
from typing import List

from rl_trainer.commons import Episode
from rl_trainer.commons.experience_tuple import ExperienceTuple


class CSVEpisodeParser:

    def parse_episode(self, episode_fname: str) -> Episode:
        experience_tuples: List[ExperienceTuple] = []
        with open(episode_fname, "r") as file:
            reader = csv.reader(file)
            for line in reader:
                experience_tuple = ExperienceTuple(
                    state_1=self._parse_line(line[0]),
                    action=self._parse_line(line[1]),
                    reward=float(line[2]),
                    state_2=self._parse_line(line[3]),
                    state_2_is_terminal=line[4] == "True",
                )
                experience_tuples.append(experience_tuple)
        return Episode(experience_tuples=experience_tuples)

    @staticmethod
    def _parse_line(line: str):
        nums = line.strip()[1:-1].replace("'", "").split(",")
        return [float(num) for num in nums]
