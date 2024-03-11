import os
from scipy.io import loadmat
from tqdm import tqdm
import numpy as np
import h5py
import random
import string
import matplotlib.pyplot as plt
from PIL import Image
import shutil
from Parallel import Parallel
from typing import Iterable, Any


class SpoofDetectSat:
    aval_id, aval_meas = [], {}
    new_aval_id, new_aval_meas = [], {}
    Train, CalSat, CalGround = None, None, None
    TrainSatid, CalSatid, CalGroundid = 0, 0, 0
    TestGround, TestSat = 0, 0
    TestGroundid, TestSatid = 0, 0

    nSamplePerImage = 0

    net = 0

    calResults = 0
    testResults = 0
    dsTest = 0

    rndName = 0

    sat_data_folder = "D:\\Data\\Jos\\fadeprint\\sat_data"
    terrestrial_data_folder = "D:\\Data\\Jos\\fadeprint\\terrestrial_data"

    new_sat_data_folder = "D:\\ProgrammingProjects\AI\\SpoofDetection\\data\\sat_data"
    new_terrestrial_data_folder = (
        "D:\\ProgrammingProjects\AI\\SpoofDetection\\data\\terrestrial_data"
    )

    def __init__(self):
        files = os.scandir(self.sat_data_folder)
        for entry in files:
            if entry.path.endswith(".mat") and entry.is_file():
                self.aval_id.append(entry.name[0:-4])
        self.aval_id = np.array(self.aval_id)

        files = os.scandir(self.terrestrial_data_folder)
        for i, entry in enumerate(files):
            if entry.path.endswith(".iq") and entry.is_file():
                self.aval_meas[i] = entry.name[0:-3]

        # scan the new folders
        files = os.scandir(self.new_sat_data_folder)
        for entry in files:
            if entry.path.endswith(".iq") and entry.is_file():
                self.new_aval_id.append(entry.name[0:-3])
        self.aval_id = np.array(self.aval_id)

        files = os.scandir(self.new_terrestrial_data_folder)
        for i, entry in enumerate(files):
            if entry.path.endswith(".iq") and entry.is_file():
                self.new_aval_meas[i] = entry.name[0:-3]

    def __loadMat(filename):
        with h5py.File(filename, "r") as mat_file:
            iq_data = mat_file["iq"][:]
            complex_data = iq_data["real"] + 1j * iq_data["imag"]
            complex_data = complex_data.astype(np.complex64)
            return complex_data

    def __loadIQ(filename):
        data = np.fromfile(filename, dtype=np.complex64)
        data = data.reshape((1, -1))
        return data

    def loadTrain(inp):
        (data_folder, sid) = inp
        iq_data = SpoofDetectSat.__loadMat(os.path.join(data_folder, "%i.mat" % sid))
        return iq_data

    def loadNewTrain(inp):
        (data_folder, sid) = inp
        iq_data = SpoofDetectSat.__loadIQ(os.path.join(data_folder, "%s.iq" % sid))
        return iq_data

    def loadData(
        self,
        TrainSatid: Iterable[int],
        CalSatid: int | None,
        CalGroundid: int | None,
        TestSatid: int | None,
        TestGroundid: int | None,
    ):
        self.TrainSatid = TrainSatid
        self.Train = []
        self.Train = (
            Parallel()
            .forEachTqdm(
                [(self.sat_data_folder, sid) for sid in TrainSatid],
                SpoofDetectSat.loadTrain,
                desc="Loading Train Sattelite data",
            )
            .join()
        )
        self.Train = np.concatenate(self.Train, axis=1)

        # Calibration datasets (Sat + Ground)
        print("Loading Calibration Sattelite data...")
        self.CalSatid = CalSatid
        if CalSatid is not None:
            self.CalSat = SpoofDetectSat.__loadMat(
                os.path.join(self.sat_data_folder, "%i.mat" % CalSatid)
            )

        print("Loading Calibration Ground data...")
        self.CalGroundid = CalGroundid
        if CalGroundid is not None:
            iq_data = SpoofDetectSat.__loadIQ(
                os.path.join(self.terrestrial_data_folder, "%s.iq" % CalGroundid)
            )
            # Remove the first 1e5 samples...
            self.CalGround = iq_data[:, int(1e6) :]

        # Test datasets (Sat + Ground)
        print("Loading Test Sattelite data...")
        self.TestSatid = TestSatid
        if TestSatid is not None:
            self.TestSat = SpoofDetectSat.__loadMat(
                os.path.join(self.sat_data_folder, "%i.mat" % TestSatid)
            )

        print("Loading Test Ground data...")
        self.TestGroundid = TestGroundid
        if TestGroundid is not None:
            iq_data = SpoofDetectSat.__loadIQ(
                os.path.join(self.terrestrial_data_folder, "%s.iq" % TestGroundid)
            )
            # Remove the first 1e5 samples...
            self.TestGround = iq_data[:, int(1e6) :]

    def loadDataNew(
        self,
        TrainSatid: list[str],
        CalSatid: str | None,
        CalGroundid: str | None,
        TestSatid: str | None,
        TestGroundid: str | None,
    ):
        self.TrainSatid = TrainSatid
        print("Loading Calibration Ground data...")
        if TrainSatid is not None and len(TrainSatid) > 0:
            self.Train = []
            self.Train = (
                Parallel()
                .forEachTqdm(
                    [(self.new_sat_data_folder, sid) for sid in TrainSatid],
                    SpoofDetectSat.loadNewTrain,
                    desc="Loading Train Sattelite data",
                )
                .join()
                .result()
            )
            self.Train = np.concatenate(self.Train, axis=1)

        # Calibration datasets (Sat + Ground)
        print("Loading Calibration Sattelite data...")
        self.CalSatid = CalSatid
        if CalSatid is not None:
            self.CalSat = SpoofDetectSat.__loadIQ(
                os.path.join(self.new_sat_data_folder, "%s.iq" % CalSatid)
            )

        print("Loading Calibration Ground data...")
        self.CalGroundid = CalGroundid
        if CalGroundid is not None:
            iq_data = SpoofDetectSat.__loadIQ(
                os.path.join(self.new_terrestrial_data_folder, "%s.iq" % CalGroundid)
            )
            # Remove the first 1e5 samples...
            self.CalGround = iq_data[:, int(1e5) :]

        # Test datasets (Sat + Ground)
        print("Loading Test Sattelite data...")
        self.TestSatid = TestSatid
        if TestSatid is not None:
            iq_data = SpoofDetectSat.__loadIQ(
                os.path.join(self.new_sat_data_folder, "%s.iq" % TestSatid)
            )
            self.TestSat = iq_data[:, int(1e5) :]

        print("Loading Test Ground data...")
        self.TestGroundid = TestGroundid
        if TestGroundid is not None:
            iq_data = SpoofDetectSat.__loadIQ(
                os.path.join(self.new_terrestrial_data_folder, "%s.iq" % TestGroundid)
            )
            # Remove the first 1e5 samples...
            self.TestGround = iq_data[:, int(1e5) :]

    def iqToImages(self, nSamplePerImage, name: str = None):
        self.nSamplePerImage = nSamplePerImage

        classes = ["Train", "CalSat", "CalGround", "TestSat", "TestGround"]
        rndName = ""
        if name is not None:
            rndName = name
        else:
            rndName = "".join(random.choice(string.ascii_lowercase) for _ in range(32))
        self.rndName = rndName

        for c in classes:
            C = getattr(self, c)

            print("Generating images for class: %s [%i]" % (c, nSamplePerImage))
            output_folder = "./datastore_%s/%s" % (rndName, c)
            print("Creating folder: %s" % output_folder)
            os.makedirs(output_folder, exist_ok=True)

            last = int(np.floor(C.shape[1] / nSamplePerImage))
            iq = np.reshape(C[:, 0 : (last * nSamplePerImage)], (nSamplePerImage, last))
            print("Generating images...")
            Parallel().forEachTqdm(
                [(iq[:, k], k, output_folder) for k in range(iq.shape[1])],
                SpoofDetectSat.generateImage,
                desc=f"Generating images for {c}",
            ).join()

        os.makedirs(f"./datastore_{rndName}/Calibration")
        os.makedirs(f"./datastore_{rndName}/Test")

        # Move files
        shutil.move(
            f"./datastore_{rndName}/CalSat",
            f"./datastore_{rndName}/Calibration/CalSat",
        )
        shutil.move(
            f"./datastore_{rndName}/CalGround",
            f"./datastore_{rndName}/Calibration/CalGround",
        )
        shutil.move(
            f"./datastore_{rndName}/TestSat",
            f"./datastore_{rndName}/Test/TestSat",
        )
        shutil.move(
            f"./datastore_{rndName}/TestGround",
            f"./datastore_{rndName}/Test/TestGround",
        )

    def generateImage(inp):
        (iq, k, output_folder) = inp

        real = np.real(iq)
        imag = np.imag(iq)

        qx = np.percentile(real, [1, 99])
        qy = np.percentile(imag, [1, 99])

        x = np.linspace(qx[0], qx[1], 225)
        y = np.linspace(qy[0], qy[1], 225)

        h, xedges, yedges = np.histogram2d(real, imag, bins=(x, y))
        h = h.T
        # Set boundary values to 0
        h[0, :] = 0
        h[-1, :] = 0
        h[:, 0] = 0
        h[:, -1] = 0

        # Check if any value exceeds 255
        if np.max(h) > 255:
            print("Number of pixels exceeding 255:", np.sum(h > 255))

        # Cap values to 255 and convert black to white
        h[h > 255] = 255
        h[h == 0] = 255
        H = np.stack((h, h, h), axis=-1).astype(np.uint8)

        fullFileName = "./%s/%i.tif" % (output_folder, k)
        Image.fromarray(H).save(fullFileName)

    def training(self):
        pass

    def iqToCsi(self, nSamplePerCsi):
        self.nSamplePerImage = nSamplePerCsi

        classes = ["Train", "CalSat", "CalGround", "TestSat", "TestGround"]
        rndName = "".join(random.choice(string.ascii_lowercase) for _ in range(32))
        self.rndName = rndName

        # Plot scatter plot
        plt.figure()

        for c in classes:
            C = getattr(self, c)

            if C is None or (type(C) is int and C == 0):
                print("Skipping %s" % c)
                continue

            last = int(np.floor(C.shape[1] / nSamplePerCsi))
            iq = np.reshape(C[:, 0 : (last * nSamplePerCsi)], (nSamplePerCsi, last))
            print("Generating csi...")
            csi = (
                Parallel()
                .forEachTqdm(
                    [(iq[:, k], k) for k in range(iq.shape[1])],
                    SpoofDetectSat.generateCsi,
                    desc=f"Generating csi for {c}",
                )
                .join()
                .result()
            )

            rssi, snr, freq, phase = zip(*csi)
            plt.scatter(rssi, snr, label=c, s=2)

        plt.xlabel("RSSI or Freq Offset")
        plt.ylabel("SNR or Phase Offset")
        plt.title("Frequency vs Phase Offset")
        plt.legend()
        plt.grid(True)
        plt.show()

    def generateCsi(inp):
        (iq_samples, k) = inp
        # if k > 1:
        #     return (0, 0)

        # Signal Strength (RSSI)
        rssi = np.mean(np.abs(iq_samples))  # Average magnitude of IQ samples

        # Signal-to-Noise Ratio (SNR)
        # Assuming noise is the variance of the IQ samples
        snr = np.mean(np.abs(iq_samples)) / np.std(np.abs(iq_samples))

        sample_rate = 10_000_000
        # Frequency Offset Estimation
        cir = np.correlate(iq_samples, iq_samples, mode="full")
        cir = cir[len(cir) // 2 :]  # Keep only second half for positive lags
        autocorr_fft = np.fft.fft(cir)
        autocorr_fft_abs = np.abs(autocorr_fft)

        # Find peak frequency
        peak_index = np.argmax(autocorr_fft_abs)
        frequency_offset = (peak_index / len(cir)) * (sample_rate / 2)

        # print("Estimated frequency offset:", frequency_offset)

        # Plot autocorrelation
        # plt.figure()
        # plt.plot(np.arange(len(cir)), np.real(cir), label="Real Part")
        # plt.plot(np.arange(len(cir)), np.imag(cir), label="Imaginary Part")
        # # plt.plot(np.arange(len(autocorr)), np.abs(autocorr), label="Absolute Value")
        # plt.title("Autocorrelation")
        # plt.xlabel("Delay (samples)")
        # plt.ylabel("Autocorrelation")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # Phase Offset Estimation
        phase_diff = np.angle(iq_samples[1:] / iq_samples[:-1])
        phase_offset = np.mean(phase_diff)

        # print("Estimated phase offset:", phase_offset)

        return rssi, snr, frequency_offset, phase_offset


def main():
    ground = [
        "mar-11-1",
        "mar-11-2",
    ]

    sds = SpoofDetectSat()

    random_ids_1 = np.random.permutation(len(sds.new_aval_id))
    random_ids_2 = np.random.permutation(len(ground))

    sds.loadDataNew(
        [sds.new_aval_id[i] for i in random_ids_1[0:-2]],
        sds.new_aval_id[random_ids_1[-2]],
        ground[0],
        sds.new_aval_id[random_ids_1[-1]],
        ground[1],
    )

    sds.iqToImages(1000, "custom_1000")
    sds.iqToImages(5000, "custom_5000")
    sds.iqToImages(10000, "custom_10000")
    sds.iqToImages(50000, "custom_50000")


def main2():
    ground = [
        "mar-11-1",
        "mar-11-2",
    ]

    sds = SpoofDetectSat()
    sds.loadDataNew(
        sds.new_aval_id[1:3],
        None,
        None,
        None,
        ground[1],
    )

    sds.iqToCsi(10000)


if __name__ == "__main__":
    import time

    time_start = time.perf_counter()
    main2()
    time_end = time.perf_counter()
    print("Execution time: %f" % (time_end - time_start))
