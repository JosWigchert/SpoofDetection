import os
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
import pandas as pd


class SpoofDetectSat:
    new_aval_id, new_aval_meas = [], {}
    Satellite, Ground = None, None
    SatIds, GroundIds = 0, 0

    nSamplePerImage = 0

    net = 0

    calResults = 0
    testResults = 0
    dsTest = 0

    rndName = 0

    new_sat_data_folder = "D:\\ProgrammingProjects\\AI\\SpoofDetection\\data\\sat"
    new_terrestrial_data_folder = (
        "D:\\ProgrammingProjects\\AI\\SpoofDetection\\data\\ground"
    )

    def __init__(self):
        # scan the new folders
        files = os.scandir(self.new_sat_data_folder)
        for entry in files:
            if entry.path.endswith(".iq") and entry.is_file():
                self.new_aval_id.append(entry.name[0:-3])
        self.new_aval_id = np.array(self.new_aval_id)

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
        return data[:, int(1e6) :]

    def loadTrain(inp):
        (data_folder, sid) = inp
        iq_data = SpoofDetectSat.__loadMat(os.path.join(data_folder, "%i.mat" % sid))
        return iq_data

    def loadNewTrain(inp):
        (data_folder, sid) = inp
        iq_data = SpoofDetectSat.__loadIQ(os.path.join(data_folder, "%s.iq" % sid))
        return iq_data

    def loadDataNew(
        self,
        SatIds: list[str],
        GroundIds: list[str],
    ):
        self.SatIds = SatIds
        print("Loading Calibration Ground data...")
        if SatIds is not None and len(SatIds) > 0:
            self.Satellite = []
            self.Satellite = (
                Parallel()
                .forEachTqdm(
                    [(self.new_sat_data_folder, sid) for sid in SatIds],
                    SpoofDetectSat.loadNewTrain,
                    desc="Loading Sattelite data",
                )
                .join()
                .result()
            )
            self.Satellite = np.concatenate(self.Satellite, axis=1)

        if GroundIds is not None and len(GroundIds) > 0:
            self.Ground = []
            self.Ground = (
                Parallel()
                .forEachTqdm(
                    [(self.new_terrestrial_data_folder, sid) for sid in GroundIds],
                    SpoofDetectSat.loadNewTrain,
                    desc="Loading Ground data",
                )
                .join()
                .result()
            )
            self.Ground = np.concatenate(self.Ground, axis=1)

    def iqToImages(self, nSamplePerImage, name: str = None):
        self.nSamplePerImage = nSamplePerImage

        classes = ["Satellite", "Ground"]
        rndName = ""
        if name is not None:
            rndName = name
        else:
            rndName = "".join(random.choice(string.ascii_lowercase) for _ in range(32))
        self.rndName = rndName

        for c in classes:
            C = getattr(self, c)

            print("Generating images for class: %s [%i]" % (c, nSamplePerImage))
            output_folder = "./datastore_%s/%s" % (rndName, c.lower())
            print("Creating folder: %s" % output_folder)
            os.makedirs(output_folder, exist_ok=True)

            if C is None:
                continue

            last = int(np.floor(C.shape[1] / nSamplePerImage))
            iq = np.reshape(C[:, 0 : (last * nSamplePerImage)], (nSamplePerImage, last))
            # print("Generating images...")
            Parallel().forEachTqdm(
                [(iq[:, k], k, output_folder) for k in range(iq.shape[1])],
                SpoofDetectSat.generateImage,
                desc=f"Generating images for {c}",
            ).join()

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
        H = np.stack((h, h, h), axis=-1).astype(np.uint8)

        fullFileName = "./%s/%i.tif" % (output_folder, k)
        Image.fromarray(H).save(fullFileName)

    def iqToCsi(self, nSamplePerCsi, name: str = None):
        self.nSamplePerImage = nSamplePerCsi

        classes = ["Satellite", "Ground"]
        rndName = ""
        if name is not None:
            rndName = name
        else:
            rndName = "".join(random.choice(string.ascii_lowercase) for _ in range(32))
        self.rndName = rndName

        # Plot scatter plot
        # plt.figure()

        output = {}

        for c in classes:
            C = getattr(self, c)

            if C is None or (type(C) is int and C == 0):
                print("Skipping %s" % c)
                continue

            last = int(np.floor(C.shape[1] / nSamplePerCsi))
            iq = np.reshape(C[:, 0 : (last * nSamplePerCsi)], (nSamplePerCsi, last))
            # print("Generating csi...")
            csi = (
                Parallel()
                .forEachTqdm(
                    [iq[:, k] for k in range(iq.shape[1])],
                    SpoofDetectSat.generateCsi,
                    desc=f"Generating csi for {c}",
                )
                .join()
                .result()
            )
            output[c] = pd.DataFrame(
                csi,
                columns=[
                    "frequency_offset",
                    "channel_frequency_offset",
                    "phase_offset",
                    "phase_noise",
                    "symbol_timing_offset",
                ],
            )

        #     (
        #         frequency_offset,
        #         channel_frequency_offset,
        #         phase_offset,
        #         phase_noise,
        #         symbol_timing_offset,
        #     ) = zip(*csi)
        #     plt.scatter(channel_frequency_offset, symbol_timing_offset, label=c, s=2)

        # plt.xlabel("RSSI or Freq Offset")
        # plt.ylabel("SNR or Phase Offset")
        # plt.title("Frequency vs Phase Offset")
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        return output

    def generateCsi(iq_samples):
        """
        Estimates various channel state information (CSI) parameters from input IQ samples.

        This function estimates frequency offset, phase offset, phase noise, symbol timing offset,
        carrier frequency offset (CFO), and phase difference for refined CFO estimation from the input IQ samples.

        Parameters:
            - iq_samples (np.array(dtype.complex64)): Input IQ samples.

        Returns:
            tuple: A tuple containing the following CSI parameters:
                - frequency_offset (float): Estimated frequency offset in Hertz.
                - channel_frequency_offset (float): Estimated carrier frequency offset in Hertz.
                - phase_offset (float): Estimated phase offset in radians.
                - phase_noise (float): Estimated phase noise.
                - symbol_timing_offset (int): Estimated symbol timing offset.
        """
        sample_rate = 10_000_000

        # Frequency Offset Estimation
        autocorr = np.correlate(iq_samples, iq_samples, mode="full")
        cir = autocorr[len(autocorr) // 2 :]  # Keep only second half for positive lags
        autocorr_fft = np.fft.fft(cir)
        autocorr_fft_abs = np.abs(autocorr_fft)

        # Find peak frequency
        peak_index = np.argmax(autocorr_fft_abs)
        frequency_offset = (peak_index / len(cir)) * (sample_rate / 2)

        # Phase Offset Estimation
        phase_diff = np.angle(iq_samples[1:] / iq_samples[:-1])
        phase_offset = np.mean(phase_diff)

        # CFO Estimation
        downconverted_samples = iq_samples * np.exp(
            -1j * 2 * np.pi * frequency_offset * np.arange(len(iq_samples))
        )

        phase_diff_cfo = np.angle(
            downconverted_samples[1:] * np.conj(downconverted_samples[:-1])
        )
        channel_frequency_offset = np.mean(phase_diff_cfo) / (
            2 * np.pi * (1 / sample_rate)
        )

        # Phase Noise Estimation (Phase variation over time)
        phase_noise = np.var(phase_diff)

        # Symbol Timing Offset Estimation (Based on autocorrelation of the received signal)
        max_autocorr_index = np.argmax(autocorr)
        symbol_timing_offset = len(iq_samples) // 2 - max_autocorr_index

        return (
            float(frequency_offset.real),
            float(channel_frequency_offset.real),
            float(phase_offset.real),
            float(phase_noise.real),
            float(symbol_timing_offset.real),
        )


def main():
    ground = [
        "apr1-1",
        "apr1-2",
        # "apr1-3",
        # "apr1-4",
        # "apr1-5",
        # "apr1-6",
        # "apr1-7",
    ]

    sds = SpoofDetectSat()

    random_ids_1 = np.random.permutation(len(sds.new_aval_id))
    random_ids_2 = np.random.permutation(len(ground))

    sds.loadDataNew(
        sds.new_aval_id,
        ground,
    )

    arr = sds.iqToCsi(5000, "autoencoder")
    keys = arr.keys()
    for key in keys:
        arr[key].to_csv(f"data/{key}.csv", index=False)


def train_csi():
    from csianomaly import get_datasets, train_model, evaluate_model
    import json

    csi = {}
    if os.path.exists("data.json"):
        with open("data.json") as f:
            csi = json.load(f)
    else:
        ground = [
            "mar-11-1",
            "mar-11-2",
        ]

        sds = SpoofDetectSat()
        sds.loadDataNew(
            sds.new_aval_id[2:],
            sds.new_aval_id[0],
            ground[0],
            sds.new_aval_id[1],
            ground[1],
        )

        csi = sds.iqToCsi(10000)
        print(csi.keys())

        with open("data.json", "w") as f:
            json.dump(csi, f)

    train_loader, val_loader, test_loader_normal, test_loader_anomalous = get_datasets(
        csi["Train"], csi["TestSat"], csi["TestGround"], csi["CalSat"], csi["CalGround"]
    )

    train_model(train_loader, val_loader)
    print("Test Loss (Normal Data):", evaluate_model(test_loader_normal))
    print("Test Loss (Anomalous Data):", evaluate_model(test_loader_anomalous))


if __name__ == "__main__":
    import time

    time_start = time.perf_counter()
    main()
    time_end = time.perf_counter()
    print("Execution time: %f" % (time_end - time_start))
