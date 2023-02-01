import numpy as np

# Part A
r_a = np.array([2, 5, 11]) # ft
F_a = np.array([0, 44, -60]) # lbs

r_x_F_a = np.cross(r_a, F_a)

print('r X F:', r_x_F_a, 'lb-ft')

# Part B
r_b = np.array([1.4, 3, 5.7]) # m
F_b = np.array([13, 25, -8]) # kN

r_x_F_b = np.cross(r_b, F_b)

print('r X F:', r_x_F_b, 'kN-m')

# Part C
r_c = np.array([-10, 15, -5]) # m
F_c = np.array([500, -250, -300]) # N

r_x_F_c = np.cross(r_c, F_c)

print('r X F:', r_x_F_c, 'N-m')