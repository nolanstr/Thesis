#include <stdio.h>
#include <math.h>

typedef struct{
    float i,j,k;
    }Vector;

Vector vecAdd(Vector a, Vector b)
{
    Vector c = {a.i + b.i, a.j + b.j, a.k + b.k};
    return c;
}

Vector vecSubtract(Vector a, Vector b)
{
    Vector c = {a.i - b.i, a.j - b.j, a.k - b.k};
    return c;
}

float dotProduct(Vector a, Vector b)
{
    return a.i*b.i + a.j*b.j + a.k*b.k;
}

float vecMag(Vector a)
{
    return pow(a.i*a.i + a.j*a.j + a.k*a.k, 0.5);
}

Vector crossProduct(Vector a,Vector b)
{
    Vector c = {a.j*b.k - a.k*b.j, 
                a.k*b.i - a.i*b.k, 
                a.i*b.j - a.j*b.i};
    return c;
}

void printVector(Vector a)
{
    printf("( %f, %f, %f)\n", a.i, a.j, a.k);
}

int main(){
    double F, t, r, m_x;
    
    // Part-a using scalar
    F = 100;
    t = 60*M_PI/180;
    r = 250;
    
    m_x = -F*sin(t)*r;
    
    // Part-b vector approach
    Vector o = {0, 0, 0};
    Vector R = {0, 250, 0};
    Vector f = {0, F*cos(t), -F*sin(t)};
    
    // Unit vector
    Vector x = {1, 0, 0};
    
    // Position vector
    Vector d = vecSubtract(R, o);
    
    // Calculate moment
    Vector M = crossProduct(R, f);
    
    // Vector magnitude
    float m;
    m = vecMag(M);
    
    printf("\n Part a (Scalar approach)\n %f (N-mm)\n", m_x);
    printf("\n Part b (Vector approach)\n r x F = "); printVector(M);
    printf("\n Magnitude = %f (N-mm)\n ", m);
    printf("\n Magnitude of M about x-axis = %f (N-mm)\n", dotProduct(M, x));
    return 0;
}