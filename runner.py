from subprocess import Popen, PIPE
from threading import Thread
from time import sleep
from signal import SIGTERM
import os
import argparse


class ProcKiller(Thread):
    def __init__(self, proc: Popen, time_limit):
        super(ProcKiller, self).__init__(target=self.run)
        self.proc = proc
        self.time_limit = time_limit

    def run(self):
        sleep(self.time_limit)
        self.kill()

    def kill(self):
        self.proc.terminate()
        print("Process killed", end="\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process launcher with timeout")
    parser.add_argument("command", nargs="+", help="Command to execute")
    parser.add_argument(
        "--time-limit", type=int, default=2, help="Time limit for the process"
    )
    args = parser.parse_args()

    cmd = args.command

    while True:
        try:
            print("Starting process: ")
            p = Popen(cmd)
            t = ProcKiller(p, args.time_limit)
            t.start()
            p.communicate()
        except KeyboardInterrupt:
            print("Exiting...")
            t.kill()
            break
