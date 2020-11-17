import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Initialize the fig and axis
# fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(6, 6))

gridsize = (8, 2)
fig = plt.figure(figsize=(6, 6))
ax1 = plt.subplot2grid(gridsize, (0,0), colspan=2, rowspan=5)
ax2 = plt.subplot2grid(gridsize, (5,0), colspan=1, rowspan=2)
ax3 = plt.subplot2grid(gridsize, (5,1), colspan=1, rowspan=2)

# Initialize camera position
ax1.plot(0, 'r*', markersize=10, label = 'Camera')

# Initialize line
line0, = ax1.plot([], [], 'b', label='Object-0', lw=2)
line1, = ax1.plot([], [], 'r', label='Object-1', lw=2)
# line2, = ax1.plot([], [], 'g', label='Object-2', lw=2)
# line3, = ax1.plot([], [], 'gray', label='Object-3', lw=2)
lines = [line0, line1]
# lines = [line0]

# Axis label and range
ax1.set(xlim=(-20, 20), ylim=(-0.5, 50))
ax1.set_xlabel('x (meter)')
ax1.set_ylabel('z (meter)')
ax1.set_title('Trajectory')
ax1.legend(loc = "upper left")
# x_range = np.linspace(-2, 2, 50)
# zmin = [4.55] * len(x_range)
# zmax = [9.55] * len(x_range)
# ax1.plot(x_range, zmin, color = 'green', linestyle = 'dashed')
# ax1.plot(x_range, zmax, color = 'green', linestyle = 'dashed')
# z_range = np.linspace(4.55, 9.55, 50)
# xmin = [-2]*len(z_range)
# xmax = [2]*len(z_range)
# ax1.plot(xmin, z_range, color = 'green', linestyle = 'dashed')
# ax1.plot(xmax, z_range, color = 'green', linestyle = 'dashed')

temp = 0

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def animate_trajectory(i):
    try:
        data = pd.read_csv("C:/Users/HP/Documents/GitHub/yolov4/output/r.txt", sep=" ", header=None)

        data = data.drop([9], axis=1)
        # data = data.drop([8], axis=1) # drop frame

        data.columns = ["time", "id", "x", "z", "vx", "vz", "dx", "dz", "frame"]
        data = data.tail(50)
        data = data.loc[data["dz"] != 0]

        # print(data.iloc[:5])

        # Eliminate row if id != 0
        data_0 = data.loc[data[["id"]].eq(0, axis='columns').to_numpy()]
        # We only select the 25 latest data
        # data_0 = data_0.tail(25)

        # Eliminate row if id != 1
        data_1 = data.loc[data[["id"]].eq(1, axis='columns').to_numpy()]
        # We only select the 25 latest data
        # data_1 = data_1.tail(25)

        # # Eliminate row if id != 1
        # data_2 = data.loc[data[["id"]].eq(2, axis='columns').to_numpy()]
        # # We only select the 25 latest data
        # data_2 = data_2.tail(25)
        #
        # # Eliminate row if id != 1
        # data_3 = data.loc[data[["id"]].eq(3, axis='columns').to_numpy()]
        # # We only select the 25 latest data
        # data_3 = data_3.tail(25)

        # Extract important information to plot
        x0 = data_0[["x"]].to_numpy()
        z0 = data_0[["z"]].to_numpy()
        path_0 = np.concatenate((x0.reshape(-1, 1), z0.reshape(-1, 1)), axis=1)
        path_0 = np.transpose(path_0)

        x1 = data_1[["x"]].to_numpy()
        z1 = data_1[["z"]].to_numpy()
        path_1 = np.concatenate((x1.reshape(-1, 1), z1.reshape(-1, 1)), axis=1)
        path_1 = np.transpose(path_1)

        # x2 = data_2[["x"]].to_numpy()
        # z2 = data_2[["z"]].to_numpy()
        # path_2 = np.concatenate((x2.reshape(-1, 1), z2.reshape(-1, 1)), axis=1)
        # path_2 = np.transpose(path_2)
        #
        # x3 = data_3[["x"]].to_numpy()
        # z3 = data_3[["z"]].to_numpy()
        # path_3 = np.concatenate((x3.reshape(-1, 1), z3.reshape(-1, 1)), axis=1)
        # path_3 = np.transpose(path_3)
        # Draw
        for lidx, garis in enumerate(lines):
            if lidx == 0:
                garis.set_data(*path_0[::, :i])
            elif lidx == 1:
                garis.set_data(*path_1[::, :i])
            # elif lidx == 2:
            #     garis.set_data(*path_2[::, :i])
            # elif lidx == 3:
            #     garis.set_data(*path_3[::, :i])

        return lines

    except:
        # print("NO FILE DETECTED ...")
        return lines


def animate_velocity(i):
    try:
        data = pd.read_csv("C:/Users/HP/Documents/GitHub/yolov4/output/r.txt", sep=" ", header=None)

        data = data.drop([9], axis=1)
        data = data.drop([8], axis=1)  # drop frame

        data.columns = ["time", "id", "x", "z", "vx", "vz", "dx", "dz"]

        # Eliminate row if id != 0
        data_0 = data.loc[data[["id"]].eq(0, axis='columns').to_numpy()]

        # Extract velocity information to plot
        sbx_0 = data_0[["time"]].to_numpy().reshape(-1, 1)
        vx_0 = data_0[["vx"]].to_numpy().reshape(-1, 1)
        vz_0 = data_0[["vz"]].to_numpy().reshape(-1, 1)


        # Eliminate row if id != 1
        data_1 = data.loc[data[["id"]].eq(1, axis='columns').to_numpy()]

        # Extract velocity information to plot
        sbx_1 = data_1[["time"]].to_numpy().reshape(-1, 1)
        vx_1 = data_1[["vx"]].to_numpy().reshape(-1, 1)
        vz_1 = data_1[["vz"]].to_numpy().reshape(-1, 1)


        # # Eliminate row if id != 2
        # data_2 = data.loc[data[["id"]].eq(2, axis='columns').to_numpy()]
        #
        # # Extract velocity information to plot
        # sbx_2 = data_2[["time"]].to_numpy().reshape(-1, 1)
        # vx_2 = data_2[["vx"]].to_numpy().reshape(-1, 1)
        # vz_2 = data_2[["vz"]].to_numpy().reshape(-1, 1)
        #
        #
        # # Eliminate row if id != 3
        # data_3 = data.loc[data[["id"]].eq(3, axis='columns').to_numpy()]
        #
        # # Extract velocity information to plot
        # sbx_3 = data_3[["time"]].to_numpy().reshape(-1, 1)
        # vx_3 = data_3[["vx"]].to_numpy().reshape(-1, 1)
        # vz_3 = data_3[["vz"]].to_numpy().reshape(-1, 1)


        # select 100 newest data
        sbx_0 = sbx_0[-100:]
        vx_0 = vx_0[-100:]
        vz_0 = vz_0[-100:]

        sbx_1 = sbx_1[-100:]
        vx_1 = vx_1[-100:]
        vz_1 = vz_1[-100:]

        # sbx_2 = sbx_2[-100:]
        # vx_2 = vx_2[-100:]
        # vz_2 = vz_2[-100:]
        #
        # sbx_3 = sbx_3[-100:]
        # vx_3 = vx_3[-100:]
        # vz_3 = vz_3[-100:]

        ax2.cla()
        ax2.plot(sbx_0, vx_0, 'tab:blue', label = "vx_0")
        ax2.plot(sbx_1, vx_1, 'tab:red', label = "vx_1")
        # ax2.plot(sbx_2, vx_2, 'tab:green', label="vx_2")
        # ax2.plot(sbx_3, vx_3, 'tab:gray', label="vx_3")
        ax2.legend(loc = "upper left")
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('(m/s)')
        ax2.set_title('Velocity x')


        ax3.cla()
        ax3.plot(sbx_0, vz_0, 'tab:blue', label = "vz_0")
        ax3.plot(sbx_1, vz_1, 'tab:red', label="vz_1")
        # ax3.plot(sbx_2, vz_2, 'tab:green', label="vz_2")
        # ax3.plot(sbx_3, vz_3, 'tab:gray', label="vz_3")
        ax3.legend(loc = "upper left")
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Velocity z')


    except:
        # print("NO FILE DETECTED ...")
        pass

# Call animation function
anim = FuncAnimation(fig, animate_trajectory, init_func = init, interval = 20, repeat_delay=5, blit=True)
anim2 = FuncAnimation(fig, animate_velocity, interval = 20, repeat_delay = 5)

# Show the plot
plt.tight_layout()
plt.show()