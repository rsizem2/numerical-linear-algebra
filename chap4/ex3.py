
import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt


# define 2x2 matrix
X = np.array([[5,2],
              [2,3]])

U,S,V = np.linalg.svd(X)

# right singular vectors
v1 = patches.Arrow(x = 0, y = 0, 
                   dx = V[0][0], 
                   dy = V[1][0],
                   width = 0.2,
                   color = "black")
v2 = patches.Arrow(x = 0, y = 0, 
                   dx = V[0][1], 
                   dy = V[1][1],
                   width = 0.2,
                   color = "black")

# unit circle (technically an ellipse)
circ = patches.Ellipse((0,0), 2, 2)

# left singular vectors, semi-axes of the image ellipse
u1 = patches.Arrow(x = 0, y = 0, 
                   dx = S[0]*U[0][0], 
                   dy = S[0]*U[1][0],
                   width = 0.2,
                   color = "black")
u2 = patches.Arrow(x = 0, y = 0, 
                   dx = S[1]*U[0][1], 
                   dy = S[1]*U[1][1],
                   width = 0.2,
                   color = "black")

# angle of rotation of the ellipse
if U[1][0] < 0:
    ang = 2*np.pi - np.arccos(U[0][0])
else:
    ang = np.arccos(U[0][0])

# ellipse representing the image of the unit circle    
ellps = patches.Ellipse((0,0), 2*S[0], 2*S[1], angle = ang*180/np.pi)



fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2)

# left subplot
ax1.add_patch(circ)
ax1.add_patch(v1)
ax1.add_patch(v2)

ax1.set(xlim = (-1.5,1.5), ylim = (-1.5,1.5),
        adjustable = 'box', aspect = 'equal')
ax1.set_title("Right Singular Vectors")


# right subplot
ax2.add_patch(ellps)
ax2.add_patch(u1)
ax2.add_patch(u2)

ax2.set(xlim = (-S[0],S[0]), ylim = (-S[0],S[0]),
        adjustable = 'box', aspect = 'equal')
ax2.set_title("Left Singular Vectors")

plt.savefig('ex3-svd.png')
plt.show()