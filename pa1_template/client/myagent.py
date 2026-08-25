import random
from typing import List, Tuple

# One play, as it is stored in history: (sign, machine, payout)
Play = Tuple[int, int, float]

N_MACHINES = 20


class StudentAgent:
    def __init__(self):
        """
        Initialize your internal state here.
        """
        pass

    def get_action(self, sign: int, history: List[List[Play]]) -> int:
        """
        Decide which machine to play.

        Args:
            sign (int): The sign lit for this play, an integer in 0..4.
            history (List): Your notebook for the month so far. It is a list of
                            nights, and each night is a list of plays. Each play
                            is a tuple (sign, machine, payout).

                            history[-1] is tonight, and it grows as the night
                            goes on. The play you are deciding right now is not
                            in it yet, so history[-1] is empty ([]) on the first
                            play of every night, and history is [[]] on the very
                            first play of the month.

        Returns:
            int: The machine to play (0 to 19).
        """

        # TODO: Implement your logic here.
        # This one ignores the sign and everything it has ever seen.

        machine = random.randrange(N_MACHINES)

        return machine  # do not change
