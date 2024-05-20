import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os
import Analyze
import string
import random

symbolsToTest = 40 #Symbols to test for finding startFrame


def plotForTesting(iqx, iqy, txrefx, txrefy, limitAxes):
    
    nrRows = 3
    nrCols = 5

    windows = nrRows * nrCols - 1

    iqxS = np.array_split(iqx, windows)
    iqyS = np.array_split(iqy, windows)

    idx = 0
    fig, axs = plt.subplots(nrRows,nrCols)
    iqIdx = 0

    for row in range(nrRows):
        for col in range(nrCols):
            if iqIdx < len(iqxS):
                axs[row][col].scatter(iqxS[iqIdx], iqyS[iqIdx], s=0.10)
                idx += len(iqxS[0])
                axs[row][col].set_title(str(idx - len(iqxS[iqIdx])) + " : " +str(idx))
                if limitAxes:
                    axs[row][col].set_xlim([-2, 2])
                    axs[row][col].set_ylim([-2, 2])

            else:
                axs[row][col].scatter(txrefx, txrefy, s=0.10)
                if limitAxes:
                    axs[row][col].set_xlim([-2, 2])
                    axs[row][col].set_ylim([-2, 2])
            iqIdx += 1
            

    # axs[0][0].scatter(iqxS[0], iqyS[0], s=0.10)
    # idx += len(iqxS[0])
    # axs[0][0].set_title(str(idx - len(iqxS[0])) + " : " +str(idx))

    # axs[0][1].scatter(iqxS[1], iqyS[1], s=0.10)
    # idx += len(iqxS[1])
    # axs[0][1].set_title(str(idx - len(iqxS[1])) + " : " +str(idx))

    # axs[0][2].scatter(iqxS[2], iqyS[2], s=0.10)
    # idx += len(iqxS[2])
    # axs[0][2].set_title(str(idx - len(iqxS[2])) + " : " +str(idx))

    # axs[1][0].scatter(iqxS[3], iqyS[3], s=0.10)
    # idx += len(iqxS[3])
    # axs[1][0].set_title(str(idx - len(iqxS[3])) + " : " +str(idx))

    # axs[1][1].scatter(iqxS[4], iqyS[4], s=0.10)
    # idx += len(iqxS[4])
    # axs[1][1].set_title(str(idx - len(iqxS[4])) + " : " +str(idx))

    # axs[1][2].scatter(iqxS[5], iqyS[5], s=0.10)
    # idx += len(iqxS[5])
    # axs[1][2].set_title(str(idx - len(iqxS[5])) + " : " +str(idx))

    # axs[2][0].scatter(iqxS[6], iqyS[6], s=0.10)
    # idx += len(iqxS[6])
    # axs[2][0].set_title(str(idx - len(iqxS[6])) + " : " +str(idx))

    # axs[2][1].scatter(iqxS[7], iqyS[7], s=0.10)
    # idx += len(iqxS[7])
    # axs[2][1].set_title(str(idx - len(iqxS[7])) + " : " +str(idx))
    # axs[2][2].scatter(txrefx, txrefy, s=0.10)

    # fig.set_figwidth(3)
    # fig.set_figheight(6)
    plt.show()
    # plt.savefig(root + rxFile + "_Clean.png")

def searchSequenceInTx(rxX, txRefX, RxFromIndex):
    #Check if symbols can be found
    if RxFromIndex < 0:
        print("Error: Search Sequence in TX, Invalid Input")
        exit(1)

    for txSymbolIdx in range(len(txRefX)-symbolsToTest):
        if (txRefX[txSymbolIdx:txSymbolIdx+symbolsToTest] == rxX[RxFromIndex:RxFromIndex+symbolsToTest]).all():
            print("Found TX Start Symbol: " + str(txSymbolIdx))
            return txSymbolIdx
    return -1 #If none can be found

def peekForwardTx(rxX, txX, RxFromIndex, TxFromIndex, peekElements, peekRange):
    rangeLimitTX = min(TxFromIndex+peekRange, int(len(txX)-peekElements)) - TxFromIndex
    rangeLimitRX = min(RxFromIndex+peekRange, int(len(rxX)-peekElements)) - RxFromIndex
    rangeLimit = min(rangeLimitTX, rangeLimitRX)

    if rangeLimit <= 1:
        # print("Peekaboo TX no use, 1 elem")
        return -1
 
    for i in range(rangeLimit): #Peek forward in TX, but stay within array
        if (rxX[RxFromIndex:RxFromIndex+peekElements] == txX[TxFromIndex+i:TxFromIndex+peekElements+i]).all():
            print("Peekaboo TX: " + str(i))
            return i
    return -1

def peekForwardRx(rxX, txX, RxFromIndex, TxFromIndex, peekElements, peekRange):
    rangeLimitTX = min(TxFromIndex+peekRange, int(len(txX)-peekElements)) - TxFromIndex
    rangeLimitRX = min(RxFromIndex+peekRange, int(len(rxX)-peekElements)) - RxFromIndex
    rangeLimit = min(rangeLimitTX, rangeLimitRX)

    if rangeLimit <= 1:
        # print("Peekaboo RX no use, 1 elem")
        return -1
    
    for i in range(rangeLimit): #Peek forward, but within array
        if (rxX[RxFromIndex+i:RxFromIndex+peekElements+i] == txX[TxFromIndex:TxFromIndex+peekElements]).all():
            print("Peekaboo RX " + str(i))
            return i
    return -1

def reverseSearch(rxX, txX, RxFromIndex, TxFromIndex, peekElements, peekRange):
    offset = 0
    for el in txX[TxFromIndex:TxFromIndex+peekElements]:
        rxToSearch = rxX[RxFromIndex+offset:RxFromIndex+peekRange]
        indexOfElement = np.where(rxToSearch == el)
        if not indexOfElement[0].any():
            return -1
        else:
            offset += indexOfElement[0][0]
            offset +=1
    #So we skipped RX <offset> elements to find TX<peekElements> 
        #elements within RX <peekRange>
    offset -= 1
    print("RevSearch: " + str(offset))
    return offset

def getIndexWithDecentDistance(rxUnedited, startIndex, modulation):
    maxDistance = 2
    minimumItemsWithDecentDistance = 350
    match modulation:
        case 1:
            maxDistance = 1.6 #1.43 with little safety
            minimumItemsWithDecentDistance = 350
        case 2: 
            maxDistance = 1.6 #1.48 with little safety
            minimumItemsWithDecentDistance = 350
        case 3:
            maxDistance = 1.7 #1.6 with little safety built in
            minimumItemsWithDecentDistance = 350
        case 4:
            maxDistance = 1.8 #1.79 with little safety built in
            minimumItemsWithDecentDistance = 550
        
    rxStartFinder = startIndex

    rxUX = rxUnedited[0::2]
    rxUY = rxUnedited[1::2]

    passedItemsWithDecentDistance = 0
    while rxStartFinder < len(rxUX)-symbolsToTest:
        distFromCenter = np.sqrt(np.square(rxUX[rxStartFinder]) + np.square(rxUY[rxStartFinder]))
        # print(str(rxStartFinder) + " - Dist: " + str(distFromCenter) + " X: " + str(rxUX[rxStartFinder]) + " Y: " + str(rxUY[rxStartFinder]))
        rxStartFinder += 1
        if distFromCenter < maxDistance:
            passedItemsWithDecentDistance += 1
        else: 
            passedItemsWithDecentDistance = 0

        if passedItemsWithDecentDistance >= minimumItemsWithDecentDistance: #TODO Magic Number
            rxStartFinder = rxStartFinder - passedItemsWithDecentDistance
            print("Found RX Start IDX Candidate: " + str(rxStartFinder))
            return rxStartFinder
    return -1 #No decent Index found

def findRxStartIndex(rxX, rxY, rxUnedited, txRefX, modulation, rxStartFinder):
    #First try with no change
    txStartSymbolIdx = searchSequenceInTx(rxX, txRefX, rxStartFinder)
    if txStartSymbolIdx > -1:
        return rxStartFinder, txStartSymbolIdx
    
    # rxStartFinder = getIndexWithDecentDistance(rxUnedited, rxStartFinder, modulation)
    rxStartFinder += 150

    #Naive find RX index
    while rxStartFinder < len(rxY)-symbolsToTest:
        #Then check if TX Start IDX can be found
        txStartSymbolIdx = searchSequenceInTx(rxX, txRefX, rxStartFinder)
        if txStartSymbolIdx < 0:
            print("No TX Start Idx found at RX Idx: " + str(rxStartFinder))
            rxStartFinder += 150
            # rxStartFinder = getIndexWithDecentDistance(rxUnedited, rxStartFinder, modulation)
            continue
        return rxStartFinder, txStartSymbolIdx

    print("No RX Start Idx found")
    return -1

def parseSymbols(rx, modulation):
    #Normalize RX
    if modulation == 1:
        print("Parsing BPSK")
        #For BPSK
        for idx in range(int(len(rx)/2)):
            xSample = rx[idx*2]
            if(xSample > 0):
                xSample = 1
            else:
                xSample = -1
            rx[idx*2] = xSample
            
    elif modulation == 2:
        print("Parsing QPSK")
        #For QPSK, Normalize both x and y (Q and I)
        for idx in range(len(rx)):
            component = rx[idx]
            if(component > 0):
                component = 0.70710677
            else:
                component = -0.70710677
            rx[idx] = component

    elif modulation == 3:
        #For 16-QAM, Normalize both x and y (Q and I)
        print("Parsing 16-QAM")
        for idx in range(len(rx)):
            component = rx[idx]
            if(component > 0):
                if component > 0.63245551: #Center of 0.9 and 0.3
                    component = 0.94868326
                else:
                    component = 0.31622776
            else:
                if component > -0.63245551: #Center of 0.9 and 0.3
                    component = -0.31622776
                else:
                    component = -0.94868326
            rx[idx] = component
    elif modulation == 4:
        #For 16-QAM, Normalize both x and y (Q and I)
        print("Parsing 64-QAM")
        for idx in range(len(rx)):
            component = rx[idx]
            if component > 0:
                if component > 0.61721343: #Center of 1.08 and 0.15
                    if component > 0.92582015: #Center of 1.08 and 0.77
                        component = 1.0801235
                    else:
                        component = 0.7715168
                else:
                    if component > 0.30860671: #Center of 0.46 and 0.15
                        component = 0.46291006
                    else:
                        component = 0.15430336
            else:
                if component > -0.61721343: #Exact center of 0.9 and 0.3
                    if component > -0.30860671: #Center of 0.46 and 0.15
                        component = -0.15430336
                    else:
                        component = -0.46291006
                else:
                    if component > -0.92582015: #Center of 1.08 and 0.77
                        component = -0.7715168
                    else:
                        component = -1.0801235
            rx[idx] = component
    return rx

def findMaxEuclidianDistance(IQx, IQy):
    rxStartFinder = 0
    distances = []
    while rxStartFinder < int(len(IQx)):
        distFromCenter = np.sqrt(np.square(IQx[rxStartFinder]) + np.square(IQy[rxStartFinder]))
        # print(str(rxStartFinder) + " - Dist: " + str(distFromCenter) + " X: " + str(rxUX[rxStartFinder]) + " Y: " + str(rxUY[rxStartFinder]))
        rxStartFinder += 1
        distances.append(distFromCenter)

    print("Max Euclidian Distance: " + str(np.max(distances)))

def getModulationReferenceXandY(modulation):
    match modulation:
        case 1:
            print(">Analyzing BPSK<")
            refFileName = "tx_bpsk.iq"
        case 2:
            print(">Analyzing QPSK<")
            refFileName = "tx_qpsk.iq"
        case 3:
            print(">Analyzing 16-QAM<")
            refFileName = "tx_16qam.iq"
        case 4:
            print(">Analyzing 64-QAM<")
            refFileName = "tx_64qam.iq"
        case _:
            print("Wrong Modulation Format")
            exit
    
    txReference = np.fromfile(open(refFileName), dtype=np.float32)
    txRefX = txReference[0::2]
    txRefY = txReference[1::2]
    return txRefX, txRefY

def generateCILowerUpper(CI, result):
    if CI == 0:
        return float('NaN'), float('NaN')
    
    lower = CI
    upper = CI
    if CI + result > 1:
        upper = 1 - result
    if result - CI < 0:
        lower = result
    return lower, upper

def measurementStitcher(root):
    IQSamples = np.array([])
    files = [f for f in pathlib.Path().glob(root + "*.iq")]
    print("Stitching " + str(len(files)) + " files")
    for f in files:
        rxUnedited = np.fromfile(open(root + f.stem + ".iq"), dtype=np.float32)
        IQSamples = np.append(IQSamples, rxUnedited)
    
    print("Amount of Samples: " + str(len(IQSamples)/2))
    IQSamples.astype('float32').tofile(root + "stitched.iq")

    with open(root + "readme_cable.txt", "a") as myfile:
        myfile.write(str(1) + " 2 3 0.1 0.1 0 20 0 60 5.00E+06 stitched.iq" + "\n")


def generateReadmeCable(root):
    files = [f for f in pathlib.Path().glob(root + "*.iq")]
    print("Creating Readme for " +str(len(files)) + " files")
    for f in files:
        with open(root + "readme_cable.txt", "a") as myfile:
            myfile.write(str("1") + " 2 3 0.1 0.1 0 20 0 60 5.00E+06 " + f.stem + ".iq" + "\n")

def createSerList(root, modulation):
    serList = []

    # if os.path.isfile(root + "serList.npy"):
    #     serList = np.load(root + "serList.npy")

    files = [f for f in pathlib.Path().glob(root + "*.iq")]
    print("Starting Analysis for " +str(len(files)) + " files")

    for f in files:
        ser = Analyze.analyze(root, f.stem, modulation, True)
        if ser == -1: #Do not count SER if file is moved
            continue
        else:
            serList.append(ser)
    
    np.save(root + "serList", serList)

def analyzeSer(root):
    serList = []

    if os.path.isfile(root + "serList.npy"):
        serList = np.load(root + "serList.npy", allow_pickle=True)
    else:
        print("No SERList found")
        return
    
    print("Files: " + str(len(serList)))
    serMax = max(serList)
    serMin = min(serList)
    serAvg = np.average(serList)
    # p95 = np.percentile(serList, 95)
    # p5 = np.percentile(serList, 5)

    print(str(serMax) + " - " + str(serMin) + " - " + str(serAvg))
    # print(str(p95) + " - " + str(p5) )

    
def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str