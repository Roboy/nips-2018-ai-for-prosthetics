from collections import namedtuple

fields = ["initial_state", "action", "final_state"]
ExperienceTuple = namedtuple("ExperienceTuple", fields)
