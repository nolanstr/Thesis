function [A] = Least_Squares(A)
% Calculate the slope and y-intercept using matrix math
% x & y are the coordinates of points
x = A(:,1);
y = A(:,2);
Z = ones(length(x),2);
Z(:,2) = x;
% Calculate the matrix inverse for the constants of the regression
A = inv(Z'*Z)*(Z'*y);
return
end

