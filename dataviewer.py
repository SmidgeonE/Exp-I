import numpy as np

# %%
data1Dir = "8thtest.txt"
straightLineDir = "STRAIGHTLINE__Data.txt"
eightLinesDir = "8lines__Data.txt"
data1finalDir = "8thtest.npy"
straightLineFinalDir = "STRAIGHTLINE__Data.npy"
eightLinesFinalDir = "8lines__Data.npy"


def Normalise(dir, finalDir):
    dataarray=np.loadtxt(dir,skiprows=1, delimiter="\t" )
    print("First data point:")
    print(dataarray[1])

    startingxdata=dataarray[0:100,1]
    startingydata=dataarray[0:100,2]
    startingzdata=dataarray[0:100,3]

    print("drifts:\n")

    xdrift = np.mean(startingxdata)
    ydrift = np.mean(startingydata)
    zdrift = np.mean(startingzdata)

    print(xdrift, " " , ydrift , " " , zdrift)
    dataarray[:,1] -= xdrift
    dataarray[:,2] -= ydrift
    dataarray[:,3] -= zdrift

    print("Normalised data:\n")
    print(dataarray[1])

    np.save(finalDir, dataarray)


Normalise(eightLinesDir, eightLinesFinalDir)
Normalise(data1Dir, data1finalDir)
Normalise(straightLineDir, straightLineFinalDir)


# %%

def VelAndDispEuler(accel, t):
    vx = [0]
    dx = [0]

    for i in range(1,len(accel)):
        vx.append(vx[i-1] + accel[i-1]*(t[i]-t[i-1]))
        dx.append(dx[i-1] + vx[i-1]*(t[i]-t[i-1]))

    return vx, dx

# Reading Data

rawData8Lines = np.loadtxt(eightLinesDir, skiprows=1, delimiter="\t")
normalised8Lines = np.load(eightLinesFinalDir, allow_pickle=True)

rawAccel8Lines, rawVel8Lines = VelAndDispEuler(rawData8Lines[1], rawData8Lines[0])

print(rawVel8Lines)