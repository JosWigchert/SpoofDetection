import multiprocessing
from typing import Iterable, Callable, Any
from tqdm import tqdm


class Parallel:
    def __init__(self, num_processes: int = None) -> None:
        self.num_processes: int = (
            num_processes if num_processes is not None else multiprocessing.cpu_count()
        )

        self.pools = []
        self.results = None

    def forEach(
        self, iterable: Iterable[Any], function: Callable[[Any], Any]
    ) -> "Parallel":
        pool = multiprocessing.Pool(processes=self.num_processes)
        self.results = pool.map(function, iterable)
        self.pools.append(pool)

        return self

    def forEachTqdm(
        self, iterable: Iterable[Any], function: Callable[[Any], Any], **kwargs: Any
    ) -> "Parallel":
        pool = multiprocessing.Pool(processes=self.num_processes)
        self.results = list(
            tqdm(pool.imap(function, iterable), total=len(iterable), **kwargs)
        )

        self.pools.append(pool)
        return self

    def join(self) -> "Parallel":
        for pool in self.pools:
            pool.close()
            pool.join()

        return self

    def result(self):
        return self.results
