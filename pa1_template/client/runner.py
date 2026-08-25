"""Runs one month at the Nine Lives Casino.

DO NOT CHANGE ANYTHING IN THIS FILE.  Put your work in ``myagent.py``.

    python runner.py
"""

import csv
from typing import List, Tuple

import requests

from myagent import StudentAgent

SERVER_URL = "https://casino-599051367010.us-west2.run.app"
RESULT_CSV = "nightly_returns.csv"

# One play, as it is stored in history: (sign, machine, payout)
Play = Tuple[int, int, float]


class CasinoClient:
    """Talks to the casino and keeps the notebook."""

    def __init__(self, server_url: str = SERVER_URL):
        self.server_url = server_url
        # One Session for the whole month: the connection is reused instead of
        # being renegotiated on each of the 3,000 plays.
        self.http = requests.Session()

        self.session_id = 0
        self.n_machines = 0
        self.plays_per_night = 0
        self.nights = 0

        # history[i] is the list of plays of night i+1.
        self.history: List[List[Play]] = []

    def start(self) -> int:
        """Open a month.  Returns the sign lit for the very first play."""
        resp = self.http.post(f"{self.server_url}/start", json={})
        resp.raise_for_status()
        data = resp.json()

        self.session_id = data["session"]
        self.n_machines = data["n_machines"]
        self.plays_per_night = data["plays_per_night"]
        self.nights = data["nights"]
        self.history = []
        return data["sign"]

    def play(self, t: int, machine: int) -> Tuple[float, int]:
        """Play one machine.  Returns (payout, next sign)."""
        payload = {"session": self.session_id, "t": t, "machine": machine}
        resp = self.http.post(f"{self.server_url}/play", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["payout"], data["next_sign"]

    def run_month(self, agent: StudentAgent) -> List[float]:
        """Play every night of the month.  Returns the nightly returns."""
        sign = self.start()
        nightly_returns: List[float] = []
        t = 0

        for night in range(1, self.nights + 1):
            # Tonight is appended empty and grows as the night goes on, so the
            # agent can see what it has already done tonight.
            self.history.append([])
            night_return = 0.0

            for _ in range(self.plays_per_night):
                machine = agent.get_action(sign, self.history)
                self._check_machine(machine)

                payout, next_sign = self.play(t, machine)

                self.history[-1].append((sign, machine, payout))
                night_return += payout
                sign = next_sign
                t += 1

            nightly_returns.append(night_return)
            print(f"Night {night:2d}: return = {night_return:9.2f}")

        return nightly_returns

    def _check_machine(self, machine: int) -> None:
        if not isinstance(machine, int) or isinstance(machine, bool):
            raise TypeError(
                f"get_action must return an int, got {type(machine).__name__}"
            )
        if not 0 <= machine < self.n_machines:
            raise ValueError(
                f"machine must be in 0..{self.n_machines - 1}, got {machine}"
            )


def save_nightly_returns(nightly_returns: List[float], path: str) -> None:
    """Write the nightly returns out so they are easy to plot for the report."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["night", "return"])
        for night, value in enumerate(nightly_returns, start=1):
            writer.writerow([night, f"{value:.4f}"])


if __name__ == "__main__":
    client = CasinoClient()
    agent = StudentAgent()

    nightly_returns = client.run_month(agent)
    save_nightly_returns(nightly_returns, RESULT_CSV)

    final_night = len(nightly_returns)
    print()
    print(f"Month total       : {sum(nightly_returns):9.2f}")
    print(f"FINAL SCORE       : {nightly_returns[-1]:9.2f}  (Night {final_night} return)")
    print(f"Nightly returns written to {RESULT_CSV}")
