from pynput.mouse import Controller
import keyboard
import time
import numpy as np
import matplotlib.pyplot as plt

x = []
y = []
mouse = Controller()
while True:
    try:
        if keyboard.is_pressed('ctrl+alt'):
            running = True
            while running:
                time.sleep(0.03)
                x.append(mouse.position[0]*0.02)
                y.append(-mouse.position[1]*0.02)
                print(mouse.position)
                if keyboard.is_pressed('shift'):
                    running = False
            # print(x)
            # print(y)
            y_des = np.array([x, y])
            np.savez('5.npz', y_des.T)
            y_des -= y_des[:, 0][:, None]
            plt.plot(y_des[0], y_des[1], 'b--', lw=2, alpha=0.7, label="mouse trajectory")
            plt.axis("equal")
            plt.xlim([int(min(y_des[0]))-1, int(max(y_des[0]))+1])
            plt.ylim([int(min(y_des[1]))-1, int(max(y_des[1]))+1])
            plt.legend()
            plt.title("Recording Mouse Trajectory")
            plt.show()
    except:
        break
