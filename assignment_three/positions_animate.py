import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd

#Picked the Hermite and bhtree 100 stars only file
#no mass seperation yet since it isn't relevant until the bridge is right and the converter is correct

pickleFile = pickle.load(open("History_DC_hermite_TC_bhtree_ClusterMass_100000.0 MSun_Radius_3.0_Cut_6.0_Flip_False_Stars_100_Timestep_0.1.p", 'rb'), fix_imports=True, encoding='latin1')


dict_data = pickleFile[1]

locations = dict_data['combined_particles_locations']
xdata = []
ydata = []
zdata = []
for list in locations:
    xdata.append(list[0])
    ydata.append(list[1])
    zdata.append(list[2])
#print(xdata[0])  

#Create a Pandas dataframe to store the locations accessing by arbitaray variable time so all the x,y,z locations for all stars for the first time step have the t = 0.0  and next time step t=1.0
t = np.array([np.ones(len(xdata[0]))*i for i in range(len(xdata))]).flatten()
df = pd.DataFrame()
df["x"] = ""
df["y"] = ""
df["z"] = ""
print(t)
#Definitely not the most effcient way but it is working

for j in range(len(xdata)):
    xlist = xdata[j]
    ylist = ydata[j]
    zlist = zdata[j]
    for i in range(len(xdata[0])):
        df = df.append({'x': xlist[i], 'y':ylist[i], 'z':zlist[i]}, ignore_index =True)
df["time"] = t
print(df[df['time']==0]['x'])
print(df[df['time']==30]['x'])

def update_graph(num):
    print(num)
    #Update all the stars 
    data=df[df['time']==num]
    graph._offsets3d = (data.x, data.y, data.z)
    #graph.set_3d_properties(data.z)
    title.set_text('3D Test, time={}'.format(num))
    return title, graph

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
title = ax.set_title('3D')

data= df[df['time']==0]
graph = ax.scatter(data.x, data.y, data.z)

ani = matplotlib.animation.FuncAnimation(fig, update_graph, len(xdata), interval=50, blit =False)

plt.show()


