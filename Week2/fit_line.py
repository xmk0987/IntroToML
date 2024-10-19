# Linear solver
def my_linfit(x,y):
    
    a = (np.sum(x * y) - np.sum(y) * np.sum(x)) / (np.sum(x ** 2) - np.sum(x) ** 2)
    b = (np.sum(y) - a * np.sum(x))  / len(x)
    return a,b

#Main

import matplotlib.pyplot as plt
import numpy as np

points_x = []
points_y = []

def record_points(event):
    if event.button == 1:
        points_x.append(event.xdata)
        points_y.append(event.ydata)
        plt.plot(event.xdata, event.ydata, 'bo')
        plt.draw()
        
    elif event.button == 3:
        if (len(points_x) > 1):
            x_point = np.array(points_x)
            y_point = np.array(points_y)            
            
            a, b = my_linfit(x_point,y_point)
            
            xp = np.linspace(min(x_point), max(x_point), 100)
            yp = a * xp + b  

           
            plt.plot(xp, yp, 'r-')
            plt.draw() 

            print(f"My fit a={a} and b={b}")
            
            
fig, ax = plt.subplots()
ax.set_title('Click to add points (Right click to stop)')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
cid = fig.canvas.mpl_connect('button_press_event', record_points)

plt.show()