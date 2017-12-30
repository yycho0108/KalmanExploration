#ifndef __MAT_OPS_H__
#define __MAT_OPS_H__

#include <cmath>
#include <cstring>
#include <iostream>
#include <cassert>
#define IDX(i,j,n) ((i)*(n)+(j))

#define BIN_OP(fn, f) \
    void fn( \
            const float* a, \
            const float* b, \
            float* o, \
            int n){ \
        if(a==o){ \
            for(int i=0; i<n; ++i){ \
                o[i] f##= b[i]; \
            } \
        }else{ \
            for(int i=0; i<n; ++i){ \
                o[i] = a[i] f b[i]; \
            } \
            return; \
        } \
    } \
    void fn( \
        const float* a, \
        float b, \
        float* o, \
        int n){ \
        for(int i=0; i<n; ++i){ \
            o[i] = a[i] f b; \
        } \
    }

#define UN_OP(fn, f) \
    void fn( \
            const float* m_i, \
            float* m_o, \
            int n){ \
        for(int i=0; i<n; ++i){ \
            m_o[i] = f(m_i[i]); \
        } \
        return; \
    }

namespace Mat{

#include "svd.h"
    inline void swap(float& a, float& b){
        float tmp=a;
        a=b;
        b=tmp;
    }

    void print(const float* mat,
            int n, int m){
        for(int i=0; i<n; ++i){
            for(int j=0; j<m; ++j){
                std::cout << mat[IDX(i,j,m)] << ' ';
            }
            std::cout << std::endl;
        }
    }

    float absmax(float* m, int n){
        // warning : computes max(abs(m))
        // not actual max.
        float mx = m[0];
        for(int i=1; i<n; ++i){
            if(fabs(m[i]) > fabs(mx))
                mx=m[i];
        }
        return mx;
    }

    bool inv(
            float* m_in,
            float* m_out,
            int n
            ){
        assert(m_in!=m_out);
        // WARNING : modifies m_in.
        // Usage : 
        // float* tmp = new float[N];
        // memcpy(tmp, m_in, N)
        // m_inv(tmp, m_out, n);

        // set to identity
        memset(m_out, 0, sizeof(float)*n*n);
        for(int i=0; i<n; ++i){
            m_out[IDX(i,i,n)] = 1.0;
        }

        for(int i=0; i<n; ++i){
            // find pivot ...
            float p_val = m_in[IDX(i,i,n)];
            int p_idx = i;
            for(int ii=i+1; ii<n; ++ii){
                float v = m_in[IDX(ii,i,n)];
                if(fabs(v) > fabs(p_val)){
                    p_val = v;
                    p_idx = ii;
                }
            }

            if(p_val == 0){
                //singular
                return false;
            }

            if(p_idx != i){
                //swap p_idx and i ...
                for(int j=0; j<n; ++j){
                    swap(m_out[IDX(i,j,n)], m_out[IDX(p_idx,j,n)]);
                    swap(m_in[IDX(i,j,n)], m_in[IDX(p_idx,j,n)]);
                }
            }

            // div row ...
            for(int j=0; j<n; ++j){
                m_out[IDX(i,j,n)] /= p_val;
                m_in[IDX(i,j,n)] /= p_val;
            }

            for(int k=0; k<n; ++k){
                if(k==i)
                    continue;
                float s = m_in[IDX(k,i,n)];
                for(int l=0; l<n; ++l){
                    m_out[IDX(k,l,n)] -= s * m_out[IDX(i,l,n)];
                    m_in[IDX(k,l,n)] -= s * m_in[IDX(i,l,n)];
                }
            }
        }
        return true;
    }


    void dot(
            const float* a, const float* b, float* o,
            int na, int nb, int nc){
        assert(a!=o);
        // zero out output matrix
        memset(o, 0, na*nb*sizeof(float));

        for(int i=0;i<na;++i){
            for(int j=0;j<nb;++j){
                for(int k=0; k<nc; ++k){
                    o[i*nb+j] += a[i*nc+k] * b[k*nb+j];
                }
            }
        }
    }

    void transpose(
            const float* m_i,
            float* m_o,
            int n_i,
            int n_j){
        for(int i=0;i<n_i;++i){
            for(int j=0;j<n_j;++j){
                m_o[IDX(j,i,n_i)] = m_i[IDX(i,j,n_j)];
            }
        }
    }

    void zero(float* m, int n){
        memset(m, 0, sizeof(float)*n);
    }

    void eye(float* m, int n_i, int n_j){
        zero(m, n_i*n_j);
        int n = n_i<n_j?n_i:n_j;
        for(int i=0; i<n; ++i){
            m[IDX(i,i,n_j)] = 1.0;
        }
    }

    bool isinf(const float* m, int n){
        for(int i=0; i<n; ++i){
            if(isinff(m[i])){
                return true;
            }
        }
        return false;
    }

    bool isnan(const float* m, int n){
        for(int i=0; i<n; ++i){
            if(isnanf(m[i])){
                return true;
            }
        }
        return false;
    }

    void pinv(float* m_i, float* m_o,
            float* ST, float* T_NN, float* T_MN,
            float* u, float* w, float* v, double* rv1,
            int n, int m,
            float eps=1e-6
            ){
        // m_i[n][m]; m_o[m][n]
        // u[n][m]
        // T_MN[m][n], T_MM[m][m]
        // u[n][n], w[...] v[m][m]
        // rv1 = float[m]
        bool trans = false;
        if(m > n){
            trans = true;

            // swap u-v names, since u[n][n] | v[m][m]
            // necessary for memory capacity's sake
            float* tmp = u;
            u = v;
            v = tmp;

            // set array transpose
            // u[j,i] = m_i[i,j], i 0-n; j 0-m
            for(int i=0; i<m; ++i){
                for(int j=0; j<m; ++j){
                    u[IDX(j,i,m)] = (i >= n)? 0.0 : m_i[IDX(i,j,m)];
                }
            }
            dsvd(u, m, n, w, v, rv1);
        }else{
            // n >= m
            for(int i=0; i<n; ++i){
                for(int j=0; j<n; ++j){
                    u[IDX(i,j,n)] = (j >= m)?0.0:m_i[IDX(i,j,m)];
                }
            }
            dsvd(u, n, m, w, v, rv1);
            //Mat::print(u, n, n);
        }

        // n > m guaranteed here
        eye(ST, m, n);
        int r = (m<n?m:n);
        for(int i=0; i<r; ++i){
            ST[IDX(i,i,n)] = (w[i] < eps? 0.0 : 1 / w[i]);
        }

        // TODO : handle here
        // A = UWV^T, A^T = VW^TU^T
        if(trans){
            // m > n
            dot(u, ST, T_MN, m, n, m);
            transpose(v, T_NN, n, n); // T_NN = v.T
            dot(T_MN, T_NN, m_o, m, n, n);
        }else{
            // n > m
            dot(v, ST, T_MN, m, n, m);
            transpose(u, T_NN, n, n); // T_NN = u.T
            dot(T_MN, T_NN, m_o, m, n, n);
        }
    }

    void lerp(float* m1, float* m2, float* mo, int n, float a){
        for(int i=0; i<n; ++i){
            mo[i] = m1[i]*a + m2[i]*(1-a);
        }
    }

    BIN_OP(add, +);
    BIN_OP(sub, -);
    BIN_OP(mul, *);
    BIN_OP(div, /);

    UN_OP(abs, fabs);
    UN_OP(neg, -);
};

#endif
