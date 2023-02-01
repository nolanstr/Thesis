clear all
clc

% Part A
r_a = 2; % ft
F_a = 8; % lbs

M_a = -r_a*F_a;

fprintf('M: %f lb-ft\n', M_a)

% Part B
r_b = 1.5; % ft
F_b = 20; % lbs

M_b = -r_b*F_b;

fprintf('M: %f lb-ft\n', M_b)

% Part C
r_c_1 = 4; % m
F_c_1 = 75; % N

r_c_2 = 1.5; % m
F_c_2 = 50; % N

M_c = -r_c_1*F_c_1 + -r_c_2*F_c_2;

fprintf('M: %f N-m\n', M_c)