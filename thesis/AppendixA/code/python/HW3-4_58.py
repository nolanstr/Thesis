import numpy as np

# Sum moments about the x-axis
F = 100 # N
t = 60*np.pi/180 # degrees
r = 250 # mm

m_x = -F*np.sin(t)*r

print(m_x)

# Vector cross product approach
o = np.array([0, 0, 0])
r = np.array([0, 250, 0]) # mm

F_vec = np.array([0, F*np.cos(t), -F*np.sin(t)])

# Position vectors
r_vec = r - o

M = np.cross(r_vec, F_vec)

M_mag = np.dot(M, M)**0.5
print(f'Magnitude of M {M_mag} N-mm')

x_hat = np.array([1, 0, 0])
print(f'Magnitude of M about x-axis: {np.dot(M, x_hat)} N-mm')