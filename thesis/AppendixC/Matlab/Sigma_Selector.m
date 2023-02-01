% Sigma selection parameter
% Christopher Creveling

close all
clear
clc

[file_name_root, dirname] = uigetfile('*.tif');
info = imfinfo(file_name_root);
% Gathers the resolution from the image data
resolution = info.XResolution; 

line_width = 0.026; % Micron length
U =  204; % Image upper intensity value (background)
P = 160; % Pixel intensity for the contrast value

% line_width = input('Max of  four line width measurements (Microns)\n');
fprintf('Resolution %f (pixels/micron)\n', resolution);

fprintf('Line width %f (microns)\n', line_width);
% resolution = 623.1429; % conversion between length and pixels

L = line_width*resolution; %Line width in pixels
fprintf('Line width %f (pixels)\n', L);
w = L/2; % width of a line in pixels
sigma = w/sqrt(3) + 0.4; % calculated sigma value
% sigma = 3.1
fprintf('Sigma = %f\n', sigma)

% sigma = 3.8; % approximate value

fprintf('U --- %d\n', U);
fprintf('P --- %d\n', P);
% Contrast (difference between upper and selected pixel intensity values)
h = U - P;
%h = 42;
fprintf('h = %d\n', h)

% First derivative of the gaussian kernel [Equation 4] - 1D
g_p1Dx = @(x, sigma)-x/(sqrt(2*pi)*sigma^3)*exp(-(x^2)/(2*sigma^2)); 
% Second directional derivative approximation [Equation 8]
rb_pp1Dx = @(x) h*(g_p1Dx(x + w, sigma) - g_p1Dx(x - w, sigma)); 
% Evaluate the second order approximation at zero to find out the upper 
% threshold value 1D
fprintf('1D upper threshold approximation is %f\n', abs(rb_pp1Dx(0))) 

% First derivative of the 2D gaussian kernel [Equation 4]
% g_p2Dx = @(x, y, sigma)-x/(2*pi*sigma^4)*exp(-(x^2 + y^2)/(2*sigma^2));
% First derivative of the 2D gaussian kernel [Equation 4]
% g_p2Dy = @(x, y, sigma)-y/(2*pi*sigma^4)*exp(-(x^2 + y^2)/(2*sigma^2)); 
% Second directional derivative approximation [Equation 8]
% rb_pp2D = @(x, y) h*(g_p2Dx(x + w, y, sigma) - ...
%     g_p2Dx(x - w, y, sigma) + g_p2Dy(x, y + w, sigma) - ...
%     g_p2Dy(x, y - w, sigma));
% Evaluate the second order approximation at zero to find out the upper 
% threshold value 1D
% fprintf('2D upper threshold approximation is %f\n', abs(rb_pp2D(0, 0))) 
% s = 0.006:0.001:0.03; % Range of sigma values
% 
% for i = 1:length(s)
%     H(i) = abs(h*(g_p1Dx(0 + w, s(i)) - g_p1Dx(0 - w, s(i))));
% end
% H';

