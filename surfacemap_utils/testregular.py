import scipy
import numpy as np
from scipy.interpolate import RegularGridInterpolator as Interpolator

fz = lambda x, y: x**2 + np.sin( y )

x_array = np.linspace(0,1)
y_array = np.linspace(0,3)
grid_x, grid_y = np.meshgrid(x_array, y_array)
grid_z = np.square( grid_x ) + np.sin( grid_y )
grid_z2 = np.square( grid_x ) + np.sin( grid_y )
grid_z2d = np.ndarray( ( *grid_z.shape, 2 ) )
grid_z2d[:,:,0] = grid_z
grid_z2d[:,:,1] = grid_z


myinter = Interpolator( (x_array, y_array), grid_z )
myinter2 = Interpolator( (x_array, y_array), grid_z2d )
print( myinter((0,1)), myinter2((0,1)))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure()
ax = fig.gca( projection='3d' )
surf = ax.plot_wireframe( grid_x, grid_y, grid_z )
#surf = ax.plot_surface( grid_x, grid_y, grid_z, cmap=cm.coolwarm, \
#                                        linewidth=0, antialiased=False )

#fig = plt.figure()
#ax = fig.gca( projection='3d' )
x_test = np.linspace(0,1)
y_test = np.linspace(0,3)
z_test = myinter( (x_test, y_test) )
ax.plot( x_test, y_test, z_test )
plt.show()
