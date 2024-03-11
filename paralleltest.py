from Parallel import Parallel
import time
import random


class Square:
    def square(self, x):
        print("Sleeping for %f seconds" % (random.random() * 2))
        time.sleep(random.random() * 2)
        return x * x

    def main(self):
        numbers = list(range(100))
        squared_numbers_with_tqdm = Parallel().forEachTqdm(
            numbers, self.square, desc="Squaring numbers"
        )
        print("Squared numbers with tqdm:", squared_numbers_with_tqdm)


if __name__ == "__main__":
    Square().main()
