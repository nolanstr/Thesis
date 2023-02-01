import numpy as np

# Given
F1 = 35 # lbs
F2 = 45 # lbs
d1 = 2.75 # ft
d2 = 1.25 # ft
t1 = 10*np.pi/180 # rad
t2 = 35*np.pi/180 # rad
b = 30*np.pi/180 # rad

# Cordinates
A = np.array([0, 0, 0])
B = np.array([d1, 0, 0])
C = np.array([d1 + d2, 0, 0])

# Position vector
AB = B - A
AC = C - A

# Force vectors
F1_vec = np.array([F1*np.sin(t1), -F1*np.cos(t1), 0])
F2_vec = np.array([-F2*np.sin(t2), -F2*np.cos(t2), 0])
print(F1_vec)
print(F2_vec)
# Moment about point A
Mb = np.cross(AB, F1_vec)
Mc = np.cross(AC, F2_vec)
print(Mb)
print(Mc)
# Resultant moment about point A
M = Mb + Mc

print(M, 'lb-ft')

# Increase moment to the max at points B & C
# Force vectors
F1_vec = np.array([F1*np.sin(0), -F1*np.cos(0), 0])
F2_vec = np.array([-F2*np.sin(0), -F2*np.cos(0), 0])

print(F1_vec)
print(F2_vec)
# Moment about point A
Mb = np.cross(AB, F1_vec)
Mc = np.cross(AC, F2_vec)
print(Mb)
print(Mc)
# Resultant moment about point A
M = Mb + Mc

print(M, 'lb-ft')