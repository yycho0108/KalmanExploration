#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MIN(x,y) ( (x) < (y) ? (x) : (y) )
#define MAX(x,y) ((x)>(y)?(x):(y))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a)) 
#ifndef IDX
#define IDX(i,j,n) ((i)*(n)+(j))
#endif

static float PYTHAG(float a, float b)
{
    float at = fabs(a), bt = fabs(b), ct, result;

    if (at > bt)       { ct = bt / at; result = at * sqrt(1.0 + ct * ct); }
    else if (bt > 0.0) { ct = at / bt; result = bt * sqrt(1.0 + ct * ct); }
    else result = 0.0;
    return(result);
}


int dsvd(float *a, int n, int m, float *w, float *v, double* rv1)
{
    int flag, i, its, j, jj, k, l, nm;
    double c, f, h, s, x, y, z;
    double anorm = 0.0, g = 0.0, scale = 0.0;
  
    if (n < m) 
    {
        fprintf(stderr, "#rows must be > #cols \n");
        return(0);
    }
  
/* Householder reduction to bidiagonal form */
    for (i = 0; i < m; i++) 
    {
        /* left-hand reduction */
        l = i + 1;
        rv1[i] = scale * g;
        g = s = scale = 0.0;
        if (i < n) 
        {
            for (k = i; k < n; k++) 
                scale += fabs(a[IDX(k,i,n)]);
            if (scale) 
            {
                for (k = i; k < n; k++) 
                {
                    a[IDX(k,i,n)] = (a[IDX(k,i,n)]/scale);
                    s += (a[IDX(k,i,n)] * a[IDX(k,i,n)]);
                }
                f = a[IDX(i,i,n)];
                g = -SIGN(sqrt(s), f);
                h = f * g - s;
                a[IDX(i,i,n)] = (f - g);
                if (i != m - 1) 
                {
                    for (j = l; j < m; j++) 
                    {
                        for (s = 0.0, k = i; k < n; k++) 
                            s += (a[IDX(k,i,n)] * a[IDX(k,j,n)]);
                        f = s / h;
                        for (k = i; k < n; k++) 
                            a[IDX(k,j,n)] += (f * a[IDX(k,i,n)]);
                    }
                }
                for (k = i; k < n; k++) 
                    a[IDX(k,i,n)] = (a[IDX(k,i,n)]*scale);
            }
        }
        w[i] = (scale * g);
    
        /* right-hand reduction */
        g = s = scale = 0.0;
        if (i < n && i != m - 1) 
        {
            for (k = l; k < m; k++) 
                scale += fabs(a[IDX(i,k,n)]);
            if (scale) 
            {
                for (k = l; k < m; k++) 
                {
                    a[IDX(i,k,n)] = (a[IDX(i,k,n)]/scale);
                    s += (a[IDX(i,k,n)] * a[IDX(i,k,n)]);
                }
                f = a[IDX(i,l,n)];
                g = -SIGN(sqrt(s), f);
                h = f * g - s;
                a[IDX(i,l,n)] = (f - g);
                for (k = l; k < m; k++) 
                    rv1[k] = a[IDX(i,k,n)] / h;
                if (i != n - 1) 
                {
                    for (j = l; j < n; j++) 
                    {
                        for (s = 0.0, k = l; k < m; k++) 
                            s += (a[IDX(j,k,n)] * a[IDX(i,k,n)]);
                        for (k = l; k < m; k++) 
                            a[IDX(j,k,n)] += (s * rv1[k]);
                    }
                }
                for (k = l; k < m; k++) 
                    a[IDX(i,k,n)] = (a[IDX(i,k,n)]*scale);
            }
        }
        anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
    }
  
    /* accumulate the right-hand transformation */
    for (i = m - 1; i >= 0; i--) 
    {
        if (i < m - 1) 
        {
            if (g) 
            {
                for (j = l; j < m; j++)
                    v[IDX(j,i,m)] = ((a[IDX(i,j,n)] / a[IDX(i,l,n)]) / g);
                    /* float division to avoid underflow */
                for (j = l; j < m; j++) 
                {
                    for (s = 0.0, k = l; k < m; k++) 
                        s += (a[IDX(i,k,n)] * v[IDX(k,j,m)]);
                    for (k = l; k < m; k++) 
                        v[IDX(k,j,m)] += (s * v[IDX(k,i,m)]);
                }
            }
            for (j = l; j < m; j++) 
                v[IDX(i,j,m)] = v[IDX(j,i,m)] = 0.0;
        }
        v[IDX(i,i,m)] = 1.0;
        g = rv1[i];
        l = i;
    }
  
    /* accumulate the left-hand transformation */
    for (i = m - 1; i >= 0; i--) 
    {
        l = i + 1;
        g = w[i];
        if (i < m - 1) 
            for (j = l; j < m; j++) 
                a[IDX(i,j,n)] = 0.0;
        if (g) 
        {
            g = 1.0 / g;
            if (i != m - 1) 
            {
                for (j = l; j < m; j++) 
                {
                    for (s = 0.0, k = l; k < n; k++) 
                        s += (a[IDX(k,i,n)] * a[IDX(k,j,n)]);
                    f = (s / a[IDX(i,i,n)]) * g;
                    for (k = i; k < n; k++) 
                        a[IDX(k,j,n)] += (f * a[IDX(k,i,n)]);
                }
            }
            for (j = i; j < n; j++) 
                a[IDX(j,i,n)] = (a[IDX(j,i,n)]*g);
        }
        else 
        {
            for (j = i; j < n; j++) 
                a[IDX(j,i,n)] = 0.0;
        }
        ++a[IDX(i,i,n)];
    }


    /* diagonalize the bidiagonal form */
    for (k = m - 1; k >= 0; k--) 
    {                             /* loop over singular values */
        for (its = 0; its < 30; its++) 
        {                         /* loop over allowed iterations */
            flag = 1;
            for (l = k; l >= 0; l--) 
            {                     /* test for splitting */
                nm = l - 1;
                if (fabs(rv1[l]) + anorm == anorm) 
                {
                    flag = 0;
                    break;
                }
                if (fabs(w[nm]) + anorm == anorm) 
                    break;
            }
            if (flag) 
            {
                c = 0.0;
                s = 1.0;
                for (i = l; i <= k; i++) 
                {
                    f = s * rv1[i];
                    if (fabs(f) + anorm != anorm) 
                    {
                        g = w[i];
                        h = PYTHAG(f, g);
                        w[i] = h; 
                        h = 1.0 / h;
                        c = g * h;
                        s = (- f * h);
                        for (j = 0; j < n; j++) 
                        {
                            y = a[IDX(j,nm,n)];
                            z = a[IDX(j,i,n)];
                            a[IDX(j,nm,n)] = (y * c + z * s);
                            a[IDX(j,i,n)] = (z * c - y * s);
                        }
                    }
                }
            }
            z = w[k];
            if (l == k) 
            {                  /* convergence */
                if (z < 0.0) 
                {              /* make singular value nonnegative */
                    w[k] = (-z);
                    for (j = 0; j < m; j++) 
                        v[IDX(j,k,m)] = (-v[IDX(j,k,m)]);
                }
                break;
            }
            if (its >= 30) {
                fprintf(stderr, "No convergence after 30,000! iterations \n");
                return(0);
            }
    
            /* shift from bottom 2 x 2 minor */
            x = w[l];
            nm = k - 1;
            y = w[nm];
            g = rv1[nm];
            h = rv1[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = PYTHAG(f, 1.0);
            f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;
          
            /* next QR transformation */
            c = s = 1.0;
            for (j = l; j <= nm; j++) 
            {
                i = j + 1;
                g = rv1[i];
                y = w[i];
                h = s * g;
                g = c * g;
                z = PYTHAG(f, h);
                rv1[j] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y = y * c;
                for (jj = 0; jj < m; jj++) 
                {
                    x = v[IDX(jj,j,m)];
                    z = v[IDX(jj,i,m)];
                    v[IDX(jj,j,m)] = (x * c + z * s);
                    v[IDX(jj,i,m)] = (z * c - x * s);
                }
                z = PYTHAG(f, h);
                w[j] = z;
                if (z) 
                {
                    z = 1.0 / z;
                    c = f * z;
                    s = h * z;
                }
                f = (c * g) + (s * y);
                x = (c * y) - (s * g);
                for (jj = 0; jj < n; jj++) 
                {
                    y = a[IDX(jj,j,n)];
                    z = a[IDX(jj,i,n)];
                    a[IDX(jj,j,n)] = (y * c + z * s);
                    a[IDX(jj,i,n)] = (z * c - y * s);

                }
            }
            rv1[l] = 0.0;
            rv1[k] = f;
            w[k] = x;
        }
    }
    return(1);
}

