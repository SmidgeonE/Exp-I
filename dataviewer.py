import numpy as np
import matplotlib.pyplot as plt
import scipy.fft

# %%
import scipy.integrate

data1Dir = "raw/8thtest.txt"
straightLineDir = "raw/STRAIGHTLINE__Data.txt"
eightLinesDir = "raw/8lines__Data.txt"

data1finalDir = "cleandata/8thtest.npy"
straightLineFinalDir = "cleandata/STRAIGHTLINE__Data.npy"
eightLinesFinalDir = "cleandata/8lines__Data.npy"

dx = 0.01


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

def velAndDispEuler(accel, t):
    velArray = [0]
    dispArray = [0]

    for i in range(1, len(accel)):
        velArray.append(velArray[i - 1] + accel[i - 1] * (t[i] - t[i - 1]))
        dispArray.append(dispArray[i - 1] + velArray[i - 1] * (t[i] - t[i - 1]))

    return velArray, dispArray


def scipyTrapezoid(accelArray):
    velArray = scipy.integrate.cumtrapz(np.array(accelArray), x=TimeData, dx=dx)
    velArray = np.append(velArray, velArray[-1])
    dispArray = scipy.integrate.cumtrapz(np.array(velArray), x=TimeData, dx=dx)
    dispArray = np.append(dispArray, dispArray[-1])


    return velArray, dispArray


# Reading Data

# rawData8Lines = np.loadtxt(eightLinesDir, skiprows=1, delimiter="\t")

#correctedData = np.load(eightLinesFinalDir, allow_pickle=True)
#correctedData = np.load(straightLineFinalDir, allow_pickle=True)
correctedData = np.load(data1finalDir, allow_pickle=True)

TimeData = correctedData[:, 0]
AccelData = [correctedData[:, 1], correctedData[:, 2], correctedData[:, 3]]
AngVelData = [correctedData[:, 4], correctedData[:, 5], correctedData[:, 6]]
VelData = [[], [], []]
DispData = [[], [], []]

# Iterate over all axes

for i in range(0, 3):
    #VelData[i], DispData[i] = velAndDispEuler(AccelData[i], TimeData)
    VelData[i], DispData[i] = scipyTrapezoid(AccelData[i])

# Reusable function to plot data in a neat way

def plotAxes(time, data, tag, ylabel):
    # plt.figure(1)
    # plt.subplot(211)
    plt.plot(time, data[0], 'r',
            time, data[1], 'g',
            time, data[2], 'b')
    plt.ylabel(ylabel)
    plt.legend([tag + '$_x$', tag + '$_y$', tag + '$_z$'], bbox_to_anchor=(1.0, 1.0))


# FFT

calcedMean = np.mean(TimeData - np.append(np.array([0]), TimeData[:len(TimeData)-1]))
rate = 1/calcedMean
n = round(rate * TimeData[-1])
Fourier = abs(scipy.fft.rfft(AccelData))
freq = scipy.fft.rfftfreq(n, calcedMean)


# Plotting all the main data

plt.figure(1)
plt.subplot(311)
plotAxes(TimeData, AccelData, 'a', 'acceleration (ms^-2)')
plt.subplot(312)
plotAxes(TimeData, VelData, 'v', 'velocity (ms^-1)')
plt.subplot(313)
plotAxes(TimeData, DispData, 'd', 'displacement (m)')

plt.figure(2)
# plt.subplot(311)
# plotAxes(TimeData, AngVelData, 'w', 'angular velocity (rad s^-1)')
plt.subplot(312)

plotAxes(freq[10:], Fourier[:, 10:], '', 'asdasd')


plt.show()