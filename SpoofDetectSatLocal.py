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

    def __init__(self):
        files = os.scandir(self.sat_data_folder)
        for entry in files:
            if entry.path.endswith(".mat") and entry.is_file():
                self.aval_id.append(int(entry.name[0:-4]))
        self.aval_id = np.array(self.aval_id)

        files = os.scandir(self.terrestrial_data_folder)
        for i, entry in enumerate(files):
            if entry.path.endswith(".iq") and entry.is_file():
                self.aval_meas[i] = entry.name[0:-3]

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
        self.Train = Parallel().forEachTqdm(
            [(self.sat_data_folder, sid) for sid in TrainSatid],
            SpoofDetectSat.loadTrain,
            desc="Loading Train Sattelite data",
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
        TrainSatid: str,
        CalSatid: int | None,
        CalGroundid: str | None,
        TestSatid: int | None,
        TestGroundid: str | None,
    ):
        self.TrainSatid = TrainSatid
        print("Loading Calibration Ground data...")
        if CalGroundid is not None:
            iq_data = SpoofDetectSat.__loadIQ(
                os.path.join(self.sat_data_folder, "%s.iq" % TrainSatid)
            )
            # Remove the first 1e5 samples...
            self.Train = iq_data[:, int(1e6) :]

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
            self.CalGround = iq_data[:, int(0) :]

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
            self.TestGround = iq_data[:, int(0) :]

    def iqToImages(self, nSamplePerImage):
        self.nSamplePerImage = nSamplePerImage

        classes = ["Train", "CalSat", "CalGround", "TestSat", "TestGround"]
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
            )

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


def main():
    ground = [
        "e534cc38457b21303d1e8c4ead21567d",
        "1740686fc144be577da7ea7864753ca1",
        "57cd30de375eb0fb15bda56d3f357b7c",
        "28d1b0ff4da1a76d6abc2d467c0f2d3f",
        "4ceca7a650c974447fcf6409762cd3a6",
    ]

    sds = SpoofDetectSat()

    random_ids_1 = np.random.permutation(len(sds.aval_id))
    random_ids_2 = np.random.permutation(len(ground))

    sds.loadDataNew(
        "total",
        sds.aval_id[random_ids_1[-2]],
        "ground",
        sds.aval_id[random_ids_1[-1]],
        "ground",
    )
    sds.iqToImages(5000)


if __name__ == "__main__":
    import time

    time_start = time.perf_counter()
    main()
    time_end = time.perf_counter()
    print("Execution time: %f" % (time_end - time_start))
