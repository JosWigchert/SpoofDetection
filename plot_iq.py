import numpy as np
import matplotlib.pyplot as plt
import random
from Parallel import Parallel


def load_iq(filename):
    data = np.fromfile(filename, dtype=np.complex64)
    data = data.reshape((-1))
    data = data[int(1e7) : int(1.1e7)]
    return data


def plot_iq(iq_data, title=""):
    # load the first 10000 iq samples and plot them in a scatter plot form -1 to +1

    plt.figure()
    plt.scatter(np.real(iq_data), np.imag(iq_data), label="IQ", s=2)
    plt.xlabel("Real")
    plt.xlim(-3, 3)
    plt.ylabel("Imag")
    plt.ylim(-3, 3)
    plt.title(title)
    plt.grid(True)
    # plt.show()
    tit = title if title != "" else random.randint(int(1e6), int(1e7))
    plt.savefig(f"plots/{tit}_iq_1.png")

    plt.figure()
    plt.scatter(np.real(iq_data[0:10000]), np.imag(iq_data[0:10000]), label="IQ", s=2)
    plt.xlabel("Real")
    plt.xlim(-3, 3)
    plt.ylabel("Imag")
    plt.ylim(-3, 3)
    plt.title(title)
    plt.grid(True)
    # plt.show()
    tit = title if title != "" else random.randint(int(1e6), int(1e7))
    plt.savefig(f"plots/{tit}_iq_2.png")


def plot_mag(mag_data, title=""):
    plt.figure()
    plt.plot(mag_data)
    plt.ylabel("Mag")
    plt.ylim(0, 3)
    plt.xlabel("Time")
    plt.title(f"{title} Avg: {np.mean(mag_data):.2f}")
    plt.grid(True)
    # plt.show()
    tit = title if title != "" else random.randint(int(1e6), int(1e7))
    plt.savefig(f"plots/{tit}.png")


def parallel_work(data):
    (path, name) = data
    iq = load_iq(path)
    mag = np.abs(iq)
    # plot_iq(iq, title="mar-17-noise-1")
    plot_iq(iq, title=name)
    plot_mag(mag, title=name)
    print(f"{name} mag avg:", np.mean(mag))


def work(path, name):
    iq = load_iq(path)
    mag = np.abs(iq)
    # plot_iq(iq, title="mar-17-noise-1")
    plot_iq(iq, title=name)
    plot_mag(mag, title=name)
    print(f"{name} mag avg:", np.mean(mag))


def main():
    data = [
        ("data/terrestrial_data/tx_qpsk_send.iq", "Send iq"),
        # ("data/terrestrial_data/mar-18-noise-1.iq", "maybe autostart"),
        # ("data/terrestrial_data/mar-18-noise-2.iq", "noise without amp"),
        # ("data/terrestrial_data/mar-18-noise-4.iq", "noise with amp"),
        # ("data/terrestrial_data/mar-18-1.iq", "data no amp"),
        # ("data/terrestrial_data/mar-18-2.iq", "data"),
        # ("data/terrestrial_data/mar-18-3.iq", "other data"),
    ]

    Parallel().forEachTqdm(data, parallel_work).join()
    # work("data/terrestrial_data/mar-18-noise-1.iq", "nothing")

    # work("data/terrestrial_data/mar-18-noise-1.iq", "maybe autostart")

    # work("data/terrestrial_data/mar-18-noise-3.iq", "noise without amp")

    # work("data/terrestrial_data/mar-18-noise-4.iq", "noise with amp")

    # work("data/terrestrial_data/mar-18-1.iq", "data no amp")

    # work("data/terrestrial_data/mar-18-2.iq", "data")

    # work("data/terrestrial_data/mar-18-3.iq", "other data")


if __name__ == "__main__":
    main()
