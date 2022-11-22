import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
import scipy.signal as sg
# %%
import scipy.integrate

data1Dir = "raw/8thtest.txt"
straightLineDir = "raw/STRAIGHTLINE__Data.txt"
eightLinesDir = "raw/8lines__Data.txt"
fiveLinesDir = "raw/15-11 5 lines 12cm_15_11_2022_Data.txt"

data1finalDir = "cleandata/8thtest.npy"
straightLineFinalDir = "cleandata/STRAIGHTLINE__Data.npy"
eightLinesFinalDir = "cleandata/8lines__Data.npy"
fiveLinesFinalDir = "cleandata/15-11 5 lines 12cm_15_11_2022_Data.npy"

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
# Normalise(fiveLinesDir, fiveLinesFinalDir)


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
#correctedData = np.load(data1finalDir, allow_pickle=True)
correctedData = np.load(fiveLinesFinalDir, allow_pickle=True)

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

def plotAxes(time, data, tag, ylabel, xlabel=''):
    # plt.figure(1)
    # plt.subplot(211)
    plt.plot(time, data[0], 'r',
            time, data[1], 'g',
            time, data[2], 'b')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend([tag + '$_x$', tag + '$_y$', tag + '$_z$'], bbox_to_anchor=(1, 1))


# FFT

calcedMean = np.mean(TimeData - np.append(np.array([0]), TimeData[:len(TimeData)-1]))
rate = 1/calcedMean
n = round(rate * TimeData[-1])
Fourier = abs(scipy.fft.rfft(AccelData))
freq = scipy.fft.rfftfreq(n, 1/rate)


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

plt.subplot(311)
plotAxes(freq[10:], Fourier[:, 10:], 'F', 'Fourier Amplitude (ms^-2)', 'Frequency (Hz)')

plt.subplot(312)
plotAxes(TimeData, AccelData, 'a', 'acceleration (ms^-2)')


# Applying Butterworth filter
#

# Indices of the largest values for the FFT, except in the first few data points
# As FFT assumes periodicity
indices = [Fourier[0,10:].argmax(), Fourier[1,10:].argmax(), Fourier[2,10:].argmax()]


# Calculates nyquist values for each axis
nyquist = np.array([freq[indices[0]], freq[indices[1]], freq[indices[2]]])/2

plt.subplot(313)
order = 5
sos = [sg.butter(order, 0.5, fs=nyquist[0], output='sos'),
       sg.butter(order, 0.5, fs=nyquist[1], output='sos'),
       sg.butter(order, 0.5, fs=nyquist[2], output='sos')]
# w, h = sg.freqs(sos)

filtered = np.array([sg.sosfilt(sos[0], AccelData[0]),
            sg.sosfilt(sos[1], AccelData[1]),
            sg.sosfilt(sos[2], AccelData[2])])

plotAxes(TimeData, filtered, 'Butterworth', 'filtered')

# Applying Savgol Filter
#


# Defines a filter windown size for savgol
# Assures it is not too big, or the movement itself will be filtered
filtersizeCoeff = 6
filtersize = round(len(TimeData)/filtersizeCoeff)
if filtersize % 2 == 0:
    filtersize -= 1


polyorder = 3
plt.figure(3)
plt.subplot(311)
plotAxes(TimeData,
         scipy.signal.savgol_filter(filtered, filtersize, polyorder),
         'Butterworth Then Savgol', 'acceleration (ms^-2)')

# plt.subplot(312)
# plotAxes(TimeData,
#          scipy.signal.savgol_filter(AccelData, filtersize, polyorder),
#          'Only Savgol Filter', 'acceleration (ms^-2)')

plt.show()