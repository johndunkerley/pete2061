# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:06:09 2019

@author: jdunke1
"""

import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

import pandas as pd
import sqlite3


conn = sqlite3.connect("DCA.db")  
cur = conn.cursor()

titleFontSize = 18
axisLabelFontSize = 15
axisNumFontSize = 13


#Question 1 
for wellID in range(1,18):#the following lines interact with the database and create dataframes which can be graphed.
    
    prodDF = pd.read_sql_query(f"SELECT time,Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)   
    
    dcaDF = pd.read_sql_query("SELECT * FROM DCAparams;", conn) 
  
    

    df1 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
    
    gasnameDF = pd.read_sql_query(f"SELECT wellID FROM DCAparams WHERE fluid = 'gas';", conn)
    
    #print(gasnameDF)
    
    oilnameDF = pd.read_sql_query(f"SELECT wellID FROM DCAparams WHERE fluid = 'oil';", conn)
    
    for wellNum in oilnameDF['wellID']:
        
        oilprodDF = pd.read_sql_query(f"SELECT time FROM Rates WHERE wellID = {wellNum};" , conn)
        
        oilcumDF = pd.read_sql_query(f"SELECT time from Rates WHERE wellID = {wellNum};",conn)
        
    for wellNum in gasnameDF['wellID']:
        
       gasprodDF = pd.read_sql_query(f"SELECT time FROM Rates WHERE wellID = {wellNum};" , conn)
       
       gascumDF = pd.read_sql_query(f"SELECT time from Rates WHERE wellID = {wellNum};",conn)

#print(gasprodDF)    
    
    
   #lines 41-52 plot production data including rates and cumulative production
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(df1['time'], df1['rate'], color="red", ls='None', marker='o', markersize=5,)
    ax2.plot(df1['time'], df1['Cum']/1000, 'b-')

    
    ax1.set_xlabel('Time, Months')
    ax1.set_ylabel('Production Rate, bopm', color='r')
    ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
    plt.title(f'Well {wellID} Monthly Production Data')
    
    plt.show()
    
#Question 2
labellist = []

for wellNum in gasnameDF['wellID']:
        
    gasprodDF['Gas Well' + str(wellNum)] = pd.read_sql_query(f"SELECT rate FROM Rates WHERE wellID = {wellNum};" , conn)
    
    scf = gasprodDF.iloc[ : , 1 : ].to_numpy()

    time = gasprodDF['time'].to_numpy()
    
    labellist.append("Gas Well " + str(wellNum))
    
    
    
    
labels = gasnameDF['wellID']
    
fig, ax = plt.subplots()
ax.stackplot(time, np.transpose(scf), labels=labellist, colors = ('y', 'r', 'b', 'g', 'm'))
ax.legend(loc='upper right')
ax.set_xlabel('Time, Months')
ax.set_ylabel('Cumulative Production, mmscf')
plt.title('Total Gas Production')
plt.show()

#Question 3
labellist = []

for wellNum in oilnameDF['wellID']:
    
   oilprodDF['Oil Well' + str(wellNum)] = pd.read_sql_query(f"SELECT rate FROM Rates WHERE wellID = {wellNum};" , conn)
    
   bopm = oilprodDF.iloc[ : , 1 : ].to_numpy()

   time = oilprodDF['time'].to_numpy()
    
   labellist.append("Oil Well " + str(wellNum))
    
    
    
    
labels = oilnameDF['wellID']
    
fig, ax = plt.subplots()
ax.stackplot(time, np.transpose(bopm), labels=labellist)
ax.legend(loc='upper right')
ax.set_xlabel('Time, Months')
ax.set_ylabel('Cumulative Production, mbbl ')
plt.title('Total Oil Production')
plt.show()

#Question 4

for wellNum in gasnameDF['wellID']:
    
    gascumDF['Gas Well ' + str(wellNum)] = pd.read_sql_query(f"SELECT cum FROM Rates WHERE wellID = {wellNum};" , conn)
 
#begin to set parameters for stacked graphs
k = 1
m = 6

indexArray = np.arange(1,m+1)    

monthList = ['POP','M2','M3','M4','M5','M6']

res = np.zeros(len(monthList))

labels = []

plotList = []

thick = 0.3

for wellNum in gasnameDF['wellID']:
    datapoint = plt.bar(gascumDF['time'][0:m], gascumDF['Gas Well '+ str(wellNum)][0:m] / 1000, thick, bottom = res)
    
    labels.append('Gas Well '+ str(wellNum))
    
    plotList.append(datapoint)
    
    plt.ylabel('Cumulative Gas Production, mscf')
    
    plt.xlabel('Months')
    
    plt.title('Cumulative Gas Production: POP to Six Months')
    
    plt.xticks(indexArray, monthList)
    
    k += 1
    
    split = gascumDF.iloc[0:6, 1:k].values
    
    
    res = np.sum(a = split, axis = 1)/1000

plt.legend(plotList, labels)

plt.show()

#Question 5
oilPL = []

for wellNum in oilnameDF['wellID']:
    
    oilcumDF['Oil Well ' + str(wellNum)] = pd.read_sql_query(f"SELECT cum FROM Rates WHERE wellID = {wellNum};" , conn)
 
l = 1

res = np.zeros(len(monthList))

oilLabels = []

for wellNum in oilnameDF['wellID']:
    
    datapoint = plt.bar(oilcumDF['time'][0:m], oilcumDF['Oil Well '+ str(wellNum)][0:m] / 1000, thick, bottom = res)
    
    oilLabels.append('Oil Well '+ str(wellNum))
    
    oilPL.append(datapoint)
    
    plt.ylabel('Cumulative Oil Production, mbbl')
    
    plt.xlabel('Months')
    
    plt.title('Cumulative Oil Production: POP to Six Months')
    
    plt.xticks(indexArray, monthList)
    
    l += 1
    
    split = oilcumDF.iloc[0:6, 1:l].values
    
    
    res = np.sum(a = split, axis = 1)/1000

plt.legend(oilPL, oilLabels,fontsize = 'x-small')

    
#Question 6:Logs
data1 = np.loadtxt("volve_logs/volve_logs/15_9-F-1B_INPUT.LAS", skiprows = 69)


DZ1, rho1 = data1[:,0], data1[:,16]

DZ1 = DZ1[np.where(rho1>0)]

rho1 = rho1[np.where(rho1>0)]

titleFontSize=22

fontSize=20

fig=plt.figure(figsize=(36,20),dpi=100)

fig.tight_layout(pad=1, w_pad=4, h_pad=2)

plt.subplot(1,6,1)

plt.grid(axis='both')

plt.plot(rho1,DZ1, color='black')

plt.title('Density v Depth', fontsize=titleFontSize, fontweight='bold')

plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')

plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')

plt.gca().invert_yaxis()

DZ1, DT1 =data1[:, 0], data1[:,8]

DZ1=DZ1[np.where(DT1>0)]

DT1=DT1[np.where(DT1>0)]

plt.subplot(1,6,2)

plt.grid(axis='both')

plt.plot(DT1,DZ1, color='blue')

plt.title('DT v Depth', fontsize=titleFontSize, fontweight='bold')

plt.xlabel('DT, us/ft', fontsize = fontSize, fontweight='bold')

plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')

plt.gca().invert_yaxis()

DZ1, DTS1 =data1[:, 0], data1[:,9]

DZ1=DZ1[np.where(DTS1>0)]

DTS1=DTS1[np.where(DTS1>0)]

plt.subplot(1,6,3)

plt.grid(axis='both')

plt.plot(DTS1,DZ1, color='red')

plt.title('DTS v Depth', fontsize = titleFontSize, fontweight='bold')

plt.xlabel('DTS, us/ft', fontsize = fontSize, fontweight='bold')

plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')

plt.gca().invert_yaxis()

DZ1, GR1 =data1[:, 0], data1[:,10]

DZ1=DZ1[np.where(GR1>0)]

GR1=GR1[np.where(GR1>0)]

plt.subplot(1,6,4)

plt.grid(axis='both')

plt.plot(GR1,DZ1, color='blue')

plt.title('GR v Depth', fontsize = titleFontSize, fontweight='bold')

plt.xlabel('GR, API', fontsize = fontSize, fontweight='bold')

plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')

plt.gca().invert_yaxis()

DZ1, NPHI1 =data1[:, 0], data1[:,12]

DZ1=DZ1[np.where(NPHI1>0)]

NPHI1=NPHI1[np.where(NPHI1>0)]

plt.subplot(1,6,5)

plt.grid(axis='both')

plt.plot(NPHI1,DZ1, color='black')

plt.title('NPHI v Depth', fontsize = titleFontSize, fontweight='bold')

plt.xlabel('NPHI, v/v', fontsize = fontSize, fontweight='bold')

plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')

plt.gca().invert_yaxis()

DZ1, CALI1 =data1[:, 0], data1[:,6]

DZ1=DZ1[np.where(CALI1>0)]

CALI1=CALI1[np.where(CALI1>0)]

plt.subplot(1,6,6)

plt.grid(axis='both')

plt.plot(CALI1,DZ1, color='m')

plt.title('Caliper v Depth', fontsize = titleFontSize, fontweight='bold')

plt.xlabel('Caliper, inch', fontsize = fontSize, fontweight='bold')

plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')

plt.gca().invert_yaxis()







data2=np.loadtxt("volve_logs/volve_logs/15_9-F-4_INPUT.LAS", skiprows=65)

DZ1, rho1= data2[:,0], data2[:,7]

DZ1= DZ1[np.where(rho1>0)]

rho1= rho1[np.where(rho1>0)]

titleFontSize=22

fontSize=20

fig=plt.figure(figsize=(36,20),dpi=100)

fig.tight_layout(pad=1, w_pad=4, h_pad=2)

plt.subplot(1,6,1)

plt.grid(axis='both')

plt.plot(rho1,DZ1, color='black')

plt.title('Density v Depth', fontsize=titleFontSize, fontweight='bold')

plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')

plt.ylabel('Depth, ft ', fontsize = fontSize, fontweight='bold')

plt.gca().invert_yaxis()

DZ1, DT1 =data2[:,0], data2[:,2]

DZ1=DZ1[np.where(DT1>0)]

DT1=DT1[np.where(DT1>0)]

plt.subplot(1,6,2)

plt.grid(axis='both')

plt.plot(DT1,DZ1, color='blue')

plt.title('DT v Depth', fontsize = titleFontSize, fontweight='bold')

plt.xlabel('DT, us/ft', fontsize = fontSize, fontweight='bold')

plt.ylabel('Depth, ft', fontsize = fontSize, fontweight='bold')

plt.gca().invert_yaxis()

DZ1, DTS1 =data2[:, 0], data2[:,3]

DZ1=DZ1[np.where(DTS1>0)]

DTS1=DTS1[np.where(DTS1>0)]

plt.subplot(1,6,3)

plt.grid(axis='both')

plt.plot(DTS1,DZ1, color='m')

plt.title('DTS v Depth', fontsize=titleFontSize, fontweight='bold')

plt.xlabel('DTS, us/ft', fontsize = fontSize, fontweight='bold')

plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')

plt.gca().invert_yaxis()

DZ1, GR1 =data2[:, 0], data2[:,4]

DZ1=DZ1[np.where(GR1>0)]

GR1=GR1[np.where(GR1>0)]

plt.subplot(1,6,4)

plt.grid(axis='both')

plt.plot(GR1,DZ1, color='blue')

plt.title('GR v Depth', fontsize = titleFontSize, fontweight='bold')

plt.xlabel('GR, API', fontsize = fontSize, fontweight='bold')

plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')

plt.gca().invert_yaxis()

DZ1, NPHI1 =data2[:, 0], data2[:,5]

DZ1=DZ1[np.where(NPHI1>0)]

NPHI1=NPHI1[np.where(NPHI1>0)]

plt.subplot(1,6,5)

plt.grid(axis='both')

plt.plot(NPHI1,DZ1, color='m')

plt.title('NPHI v Depth', fontsize = titleFontSize, fontweight='bold')

plt.xlabel('NPHI, v/v', fontsize = fontSize, fontweight='bold')

plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')

plt.gca().invert_yaxis()



data3 = np.loadtxt("volve_logs/volve_logs/15_9-F-14_INPUT.LAS", skiprows = 69)

DZ1, rho1 = data3[:,0], data3[:,9]

DZ1 = DZ1[np.where(rho1>0)]

rho1 = rho1[np.where(rho1>0)]

titleFontSize=22

fontSize=20

fig=plt.figure(figsize=(36,20),dpi=100)

fig.tight_layout(pad=1, w_pad=4, h_pad=2)

plt.subplot(1,6,1)

plt.grid(axis='both')

plt.plot(rho1,DZ1, color='black')

plt.title('Density v Depth', fontsize=titleFontSize, fontweight='bold')

plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')

plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')

plt.gca().invert_yaxis()

DZ1, DT1 =data3[:, 0], data3[:,3]

DZ1=DZ1[np.where(DT1>0)]

DT1=DT1[np.where(DT1>0)]

plt.subplot(1,6,2)

plt.grid(axis='both')

plt.plot(DT1,DZ1, color='blue')

plt.title('DT v Depth', fontsize=titleFontSize, fontweight='bold')

plt.xlabel('DT, us/ft', fontsize = fontSize, fontweight='bold')

plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')

plt.gca().invert_yaxis()

DZ1, DTS1 =data3[:, 0], data3[:,4]

DZ1=DZ1[np.where(DTS1>0)]

DTS1=DTS1[np.where(DTS1>0)]

plt.subplot(1,6,3)

plt.grid(axis='both')

plt.plot(DTS1,DZ1, color='m')

plt.title('DTS v Depth', fontsize=titleFontSize, fontweight='bold')

plt.xlabel('DTS, us/ft', fontsize = fontSize, fontweight='bold')

plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')

plt.gca().invert_yaxis()

DZ1, GR1 =data3[:, 0], data3[:,5]

DZ1=DZ1[np.where(GR1>0)]

GR1=GR1[np.where(GR1>0)]

plt.subplot(1,6,4)

plt.grid(axis='both')

plt.plot(GR1,DZ1, color='black')

plt.title('GR v Depth', fontsize=titleFontSize, fontweight='bold')

plt.xlabel('GR, API', fontsize = fontSize, fontweight='bold')

plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')

plt.gca().invert_yaxis()

DZ1, NPHI1 =data3[:, 0], data3[:,6]

DZ1=DZ1[np.where(NPHI1>0)]

NPHI1=NPHI1[np.where(NPHI1>0)]

plt.subplot(1,6,5)

plt.grid(axis='both')

plt.plot(NPHI1,DZ1, color='black')

plt.title('NPHI v Depth', fontsize=titleFontSize, fontweight='bold')

plt.xlabel('NPHI, v/v', fontsize = fontSize, fontweight='bold')

plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')

plt.gca().invert_yaxis()



    
