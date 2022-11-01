##Experiment I Starting Script##
##B. Rackauskas & M. Bird. Updated 2022 by M. Bird##

import serial
import time
import matplotlib.pyplot as plt

#establish serial connection to Arduino/GY521
ser = serial.Serial('COM3', 38400) #Baud rate 38400 Hz, COM port must match.
ser.reset_input_buffer()
ser.reset_output_buffer()
ser.flush()
for i in range(0,3):
    print(ser.readline(100).decode("utf-8","ignore").replace('\r\n','')) #print lines from GY521
    
    
res = 2**16; # sensitivity setting - must match with Arduino code
a_sen = 2* 9.81; #m/s^2
g_sen = 250; #deg/s

ax = []
ay = []
az = []
gx = []
gy = []
gz = []
t  = []

#main loop
try:
    print("Capturing data, press ctrl+C to finish")

    #obtain data
    while 1:
        s = ser.readline(100)
        #print(s)
        ss = s.decode("utf-8","ignore").replace('\r\n','').split('\t')
        ss = ss[1:]
        ax.append(ss[0])
        ay.append(ss[1])
        az.append(ss[2])
        gx.append(ss[3])
        gy.append(ss[4])
        gz.append(ss[5])
        t.append(ss[6])
        
except IndexError:
    ser.close()
    print('Stopping... Index Error. Please re-run the program')

        
except KeyboardInterrupt:
    print('Stopping...')

    #convert the read values into "physical" values
    ax = [int(i)*a_sen*2 / res for i in ax] 
    ay = [int(i)*a_sen*2 / res for i in ay] 
    az = [int(i)*a_sen*2 / res for i in az] 
    gx = [int(i)*g_sen*2 / res for i in gx]
    gy = [int(i)*g_sen*2 / res for i in gy]
    gz = [int(i)*g_sen*2 / res for i in gz]
    t = [int(tm)/1000 for tm in t]

    ser.close()    
    
    plt.figure(1)
    plt.subplot(211)
    plt.plot(t,ax,'r',t,ay,'g',t,az,'b')
    plt.ylabel('Acceleration (m/s$^2$)')
    plt.legend(['$a_x$','$a_y$','$a_z$'],bbox_to_anchor=(1.0,1.0))
        
    plt.subplot(212)
    plt.plot(t,gx,'pink',t,gy,'yellow',t,gz,'cyan')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (deg/s)')
    plt.legend(['$\omega_x$','$\omega_y$','$\omega_z$'],bbox_to_anchor=(1.0,1.0))

    plt.tight_layout()

    
    # Velocity

    vx = [0]
    vy = [0]
    vz = [0]

    dx = [0]
    dy = [0]
    dz = [0]

    for i in range(1,len(ax)):
        vx.append(vx[i-1] + ax[i-1]*(t[i]-t[i-1]))
        vy.append(vy[i-1] + ay[i-1]*(t[i]-t[i-1]))  
        vz.append(vz[i-1] + az[i-1]*(t[i]-t[i-1]))

        dx.append(dx[i-1] + vx[i-1]*(t[i]-t[i-1]))
        dy.append(dy[i-1] + vy[i-1]*(t[i]-t[i-1]))
        dz.append(dz[i-1] + vz[i-1]*(t[i]-t[i-1]))


    # Plotting

    plt.figure(2)
    plt.subplot(211)
    plt.plot(t,vx,'pink',t,vy,'yellow',t,vz,'cyan')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity ($ms^{-1}$)')
    plt.legend(['$v_x$','$v_y$','$v_z$'],bbox_to_anchor=(1.0,1.0))

    plt.figure(2)
    plt.subplot(212)
    plt.plot(t,vx,'pink',t,vy,'yellow',t,vz,'cyan')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement ($m$)')
    plt.legend(['$d_x$','$d_y$','$d_z$'],bbox_to_anchor=(1.0,1.0))
    

        
    # (optional) write the data to a text file
    timestr = time.strftime("%d_%m_%Y") 
    choice = input("Save data to .txt file? [Y/N]")
    if choice == 'Y':
        Filename = input("Enter Filename:")
        f = open(Filename + '_' + timestr + '_Data.txt','w')
        f.write('Time\tax\tay\taz\tgx\gy\gz\n')
        for i in range(len(ax)):
            f.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (t[i],ax[i],ay[i],az[i],gx[i],gy[i],gz[i]))
        f.close()
        print('File written to:' + str(f))
        print('Done')
        input()

    

    plt.show()
