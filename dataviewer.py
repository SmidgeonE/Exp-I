import numpy as np
import matplotlib.pyplot as plt

# %%
data1Dir = "raw/8thtest.txt"
straightLineDir = "raw/STRAIGHTLINE__Data.txt"
eightLinesDir = "raw/8lines__Data.txt"

data1finalDir = "cleandata/8thtest.npy"
straightLineFinalDir = "cleandata/STRAIGHTLINE__Data.npy"
eightLinesFinalDir = "cleandata/8lines__Data.npy"


def Normalise(dir, finalDir):
    dataarray = np.loadtxt(dir, skiprows=1, delimiter="\t")
    print("First data point:")
    print(dataarray[1])

    startingxdata = dataarray[0:100, 1]
    startingydata = dataarray[0:100, 2]
    startingzdata = dataarray[0:100, 3]

    print("drifts:\n")

    xdrift = np.mean(startingxdata)
    ydrift = np.mean(startingydata)
    zdrift = np.mean(startingzdata)

    print(xdrift, " ", ydrift, " ", zdrift)
    dataarray[:, 1] -= xdrift
    dataarray[:, 2] -= ydrift
    dataarray[:, 3] -= zdrift

    print("Normalised data:\n")
    print(dataarray[1])

    np.save(finalDir, dataarray)


# Normalise(eightLinesDir, eightLinesFinalDir)
# Normalise(data1Dir, data1finalDir)
# Normalise(straightLineDir, straightLineFinalDir)


# %%

def VelAndDispEuler(accel, t):
    vx = [0]
    dx = [0]

    for i in range(1, len(accel)):
        vx.append(vx[i - 1] + accel[i - 1] * (t[i] - t[i - 1]))
        dx.append(dx[i - 1] + vx[i - 1] * (t[i] - t[i - 1]))

    return vx, dx


# Reading Data

# rawData8Lines = np.loadtxt(eightLinesDir, skiprows=1, delimiter="\t")

correct8Lines = np.load(eightLinesFinalDir, allow_pickle=True)

eightLineTime = correct8Lines[:, 0]
eightLineAccel = [correct8Lines[:, 1], correct8Lines[:, 1], correct8Lines[:, 2]]
eightLineAngVel = [correct8Lines[:, 3], correct8Lines[:, 4], correct8Lines[:, 5]]
eightLineVel = [[], [], []]
eightLineDisp = [[], [], []]

for i in range(0, 3):
    eightLineVel[i], eightLineDisp[i] = VelAndDispEuler(eightLineAccel[i],
                                                        eightLineTime)


def plotAxes(time, data, tag, ylabel):
    # plt.figure(1)
    # plt.subplot(211)
    plt.plot(time, data[0], 'r',
            time, data[1], 'g',
            time, data[2], 'b')
    plt.ylabel(ylabel)
    plt.legend([tag + '$_x$', tag + '$_y$', tag + '$_z$'], bbox_to_anchor=(1.0, 1.0))

plt.figure(1)
plt.subplot(311)
plotAxes(eightLineTime, eightLineAccel, 'a', 'acceleration (ms^-2)')
plt.subplot(312)
plotAxes(eightLineTime, eightLineVel, 'v', 'velocity (ms^-1)')
plt.subplot(313)
plotAxes(eightLineTime, eightLineDisp, 'd', 'displacement (m)')

plt.show()
