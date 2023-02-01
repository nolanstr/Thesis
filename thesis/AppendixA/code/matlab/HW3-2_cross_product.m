clear all
clc

% Part A
r_a = [2, 5, 11]; % ft
F_a = [0, 44, -60]; % lbs

r_x_F_a = cross(r_a, F_a);

fmt = ['r X F: [', repmat('%g, ', 1, numel(r_x_F_a)-1), '%g] lb-ft\n'];
fprintf(fmt, r_x_F_a)

% Part B
r_b = [1.4, 3, 5.7]; % m
F_b = [13, 25, -8]; % kN

r_x_F_b = cross(r_b, F_b);

fmt = ['r X F: [', repmat('%g, ', 1, numel(r_x_F_b)-1), '%g] kN-m\n'];
fprintf(fmt, r_x_F_b)

% Part C
r_c = [-10, 15, -5]; % m
F_c = [500, -250, -300]; % N

r_x_F_c = cross(r_c, F_c);

fmt = ['r X F: [', repmat('%g, ', 1, numel(r_x_F_c)-1), '%g] N-m\n'];
fprintf(fmt, r_x_F_c)