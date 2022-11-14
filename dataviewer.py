import numpy as np

# %%
dir = "8thtest.txt"
dir1 = "STRAIGHTLINE__Data.txt"
dir2 = "8lines__Data.txt"
finaldir = "8thtestnormalised.npy"
finaldir1 = "STRAIGHTLINE__Datanormalised.npy"
finaldir2 = "8lines__Datanormalised.npy"


def Normalise(dir, finalDir):
    dataarray=np.loadtxt(dir,skiprows=1, delimiter="\t" )
    print("First data point:")
    print(dataarray[1])

    startingxdata=dataarray[0:100,1]
    startingydata=dataarray[0:100,2]
    startingzdata=dataarray[0:100,3]

    print("drifts:\n")

    xdrift=np.mean(startingxdata)
    ydrift = np.mean(startingydata)
    zdrift = np.mean(startingzdata)

    print(xdrift, " " , ydrift , " " , zdrift)
    dataarray[:,1] -= xdrift
    dataarray[:,2] -= ydrift
    dataarray[:,3] -= zdrift

    print("Normalised data:\n")
    print(dataarray[1])

    np.save(finaldir, dataarray) 

Normalise(dir1, finalDir=finaldir1)
Normalise(dir2, finalDir=finaldir2)

# %%
