import numpy as np 
import matplotlib.pyplot as plt 
def z_func(x,y):
    return np.sin(x*5) * np.cos(5*y)/5

def calculate_gradient(x,y) : 
    return np.cos(x*5) * np.cos(5*y) , -np.sin(x*5) * np.sin(5*y) 

x = np.arange(-1 , 1 , 0.05)
y = np.arange(-1 , 1 , 0.05)
X , Y = np.meshgrid(x,y) 
Z = z_func(X,Y)

current_position = (0.7 , 0.4 , z_func(0.7,0.4))
current_position1 = (0.3 , -0.2 , z_func(0.3,0.2))
current_position2 = (-0.4 , 0.7 , z_func(0.4,0.7))
alpha = 0.01
ax = plt.subplot(projection="3d" , computed_zorder=False)

for i in range(1000) : 
    dzdx , dzdy = calculate_gradient(current_position[0] , current_position[1])
    X_new , Y_new = current_position[0] - alpha*dzdx  , current_position[1] - alpha*dzdy 
    current_position = (X_new , Y_new , z_func(X_new,Y_new))
    
    dzdx , dzdy = calculate_gradient(current_position1[0] , current_position1[1])
    X_new , Y_new = current_position1[0] - alpha*dzdx  , current_position1[1] - alpha*dzdy 
    current_position1 = (X_new , Y_new , z_func(X_new,Y_new))
    
    dzdx , dzdy = calculate_gradient(current_position2[0] , current_position2[1])
    X_new , Y_new = current_position2[0] - alpha*dzdx  , current_position2[1] - alpha*dzdy 
    current_position2 = (X_new , Y_new , z_func(X_new,Y_new))
    
    ax.plot_surface(X,Y,Z,cmap='viridis' , zorder=0)
    ax.scatter(current_position[0] , current_position[1] , current_position[2] , zorder=1 , color="red")
    ax.scatter(current_position1[0] , current_position1[1] , current_position1[2] , zorder=1 , color="green")
    ax.scatter(current_position2[0] , current_position2[1] , current_position2[2] , zorder=1 , color="yellow")
    plt.pause(0.001)
    ax.clear()
