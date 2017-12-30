#ifndef __KALMAN_H__
#define __KALMAN_H__

#include <cstring>
#include "mat_ops.h"

#define DEG (M_PI / 180.0F)
#define RAD (180.0F / M_PI)

template<int DIM_X, int DIM_Z, int DIM_C=0>
class KalmanFilter{
    public:
        float t;
        float x[DIM_X]; // state
        float P[DIM_X][DIM_X]; // covariance
        float Q[DIM_X][DIM_X]; // process noise
        float R[DIM_Z][DIM_Z]; // measurement noise
        float F[DIM_X][DIM_X]; //state transition
        float H[DIM_Z][DIM_X]; // observation mapping

        float y[DIM_Z]; // error residual placeholder

        float S[DIM_Z][DIM_Z];
        float S_i[DIM_Z][DIM_Z];
        float HPH_t[DIM_Z][DIM_Z];
        float K[DIM_X][DIM_Z]; // kalman gain

        // placeholders - pred
        float F_t[DIM_X][DIM_X];
        float x_n[DIM_X];
        float P_n[DIM_X][DIM_X];
        float FP[DIM_X][DIM_X];

        // placeholders - update
        float H_t[DIM_X][DIM_Z];
        float HP[DIM_Z][DIM_X];
        float KHP[DIM_X][DIM_X];
        float PH_t[DIM_X][DIM_Z];
        float Hx[DIM_Z];
        float Ky[DIM_X];

        // ALS
        float KH[DIM_X][DIM_Z];
        float A[DIM_C-1][DIM_Z][DIM_X];
        float A_i[DIM_X][DIM_C-1];
        float IE[DIM_X][DIM_X];
        float IB[DIM_X][DIM_X];
        float T_XX[DIM_X][DIM_X];
        float E[DIM_Z][DIM_X];
        float MH[DIM_X][DIM_Z];
        float MH_a[DIM_X][DIM_Z];

        //AEKF
        float KYY[DIM_X][DIM_Z];
        float Q_k[DIM_X][DIM_X];
        float K_t[DIM_Z][DIM_X];

        float YY[DIM_Z][DIM_Z];
        float R_k[DIM_Z][DIM_Z];

        // PINV
        //  "N" = (DIM_C-1)*(DIM_Z); M = DIM_X
        float ST[DIM_X][(DIM_C-1)*DIM_Z];
        float T_NN[(DIM_C-1)*DIM_Z][(DIM_C-1)*DIM_Z];
        float T_MN[DIM_X][(DIM_C-1)*DIM_Z];
        float U[(DIM_C-1)*DIM_Z][(DIM_C-1)*DIM_Z];
        float V[DIM_X][DIM_X];
        float W[(DIM_C-1)*DIM_Z<DIM_X?(DIM_C-1)*DIM_Z:DIM_X];
        double rv1[(DIM_C-1)*DIM_Z<DIM_X?(DIM_C-1)*DIM_Z:DIM_X];
        
        // flags
        bool als;

    public:
        KalmanFilter(
                float* _P,
                float* _Q,
                float* _R,
                float* _F,
                float* _H,
                float* _x,
                bool _als=false
                ){
            memcpy(x, _x, sizeof(x));
            memcpy(P, _P, sizeof(P));
            memcpy(Q, _Q, sizeof(Q));
            memcpy(R, _R, sizeof(R));
            memcpy(F, _F, sizeof(F));
            memcpy(H, _H, sizeof(H));
            als=_als;
        }

        void predict(float* u=nullptr){
            // pred state
            Mat::dot((float*)F, (float*)x, (float*)x_n,DIM_X, DIM_X, DIM_X);
            //Mat::dot(B, u, Bu);
            //Mat::add(x_n, Bu, x)
            memcpy(x, x_n, DIM_X);
            
            // pred cov
            Mat::transpose((float*)F, (float*)F_t, DIM_X, DIM_X);
            Mat::dot((float*)F, (float*)P, (float*)FP, DIM_X, DIM_X, DIM_X); 
            Mat::dot((float*)FP, (float*)F_t, (float*)P, DIM_X, DIM_X, DIM_X);
            Mat::add((float*)P, (float*)Q, (float*)P, DIM_X*DIM_X);
        }

        void update(const float* z){
            // P = cov(e_x) = cov(H_t*y)?
            
            // residual (error)
            Mat::dot((float*)H, (float*)x, (float*)Hx, DIM_Z, 1, DIM_X);
            Mat::sub((float*)z, (float*)Hx, (float*)y, DIM_Z);

            // residual cov.
            Mat::transpose((float*)H, (float*)H_t, DIM_Z, DIM_X);
            Mat::dot((float*)H, (float*)P, (float*)HP, DIM_Z, DIM_X, DIM_X);
            Mat::dot((float*)HP, (float*)H_t, (float*)HPH_t, DIM_Z, DIM_Z, DIM_X);
            Mat::add((float*)HPH_t, (float*)R, (float*)S, DIM_Z*DIM_Z);

            // kalman gain
            Mat::inv((float*)S, (float*)S_i, DIM_Z); //S gets modified here
            Mat::dot((float*)P, (float*)H_t, (float*)PH_t, DIM_X, DIM_Z, DIM_X);
            Mat::dot((float*)PH_t, (float*)S_i, (float*)K, DIM_X, DIM_Z, DIM_Z);

            // update state
            Mat::dot((float*)K, (float*)y, (float*)Ky, DIM_X, 1, DIM_Z);
            Mat::add((float*)x, (float*)Ky, (float*)x, DIM_X);

            // Q Must be updated here, with pre-fit innovation:
            if(als){
                Mat::transpose((float*)K, (float*)K_t, DIM_X, DIM_Z);//K=XZ, K_t=ZX
                Mat::dot((float*)Ky, (float*)y, (float*)KYY, DIM_X, DIM_Z, 1);
                Mat::dot((float*)KYY, (float*)K_t, (float*) Q_k, DIM_X, DIM_X, DIM_Z);
                Mat::lerp((float*)Q, (float*)Q_k, (float*)Q, DIM_X*DIM_X, 0.1);
            }
            //post-fit residual

            Mat::dot((float*)H, (float*)x, (float*)Hx, DIM_Z, 1, DIM_X);
            Mat::sub((float*)z, (float*)Hx, (float*)y, DIM_Z);

            // R Must be updated here, with post-fit residual:
            if(als){
                Mat::dot((float*)y, (float*)y, (float*)YY, DIM_Z, DIM_Z, 1);
                Mat::add((float*)YY, (float*)HPH_t, (float*)R_k, DIM_Z*DIM_Z);
                Mat::lerp((float*)R, (float*)R_k, (float*)R, DIM_Z*DIM_Z, 0.1);
            }
            
            // update cov
            Mat::dot((float*)K, (float*)HP, (float*)KHP, DIM_X, DIM_X, DIM_Z);
            Mat::sub((float*)P, (float*)KHP, (float*)P, DIM_X*DIM_X);
        }

        void ALS_R(float* C){
            //Autocovariance Least-Squares
            
            // C = [DIM_C, DIM_Z, DIM_Z]
            Mat::eye((float*)IE, DIM_X, DIM_X);
            Mat::eye((float*)IB, DIM_X, DIM_X);

            Mat::dot((float*)K, (float*)H, (float*)KH, DIM_X, DIM_X, DIM_Z);
            Mat::sub((float*)IB, (float*)KH, (float*)IE, DIM_X*DIM_X); //IE = I-KH
            Mat::dot((float*)F, (float*)IE, (float*)T_XX, DIM_X, DIM_X, DIM_X);
            memcpy(IE, T_XX, sizeof(IE)); // IE = F*IE

            for(int i=0; i<(DIM_C-1); ++i){
                if(i > 0){
                    Mat::dot((float*)IB, (float*)IE, (float*)T_XX, DIM_X, DIM_X, DIM_X);
                    memcpy(IB,T_XX,sizeof(IB));
                }
                // H = ZX, IB = XX, E = ZX
                Mat::dot((float*)H, (float*)IB, (float*)E, DIM_Z, DIM_X, DIM_X);
                // E = ZX, F = XX, EF = ZX
                Mat::dot((float*)E, (float*)F, (float*)(&A[i]), DIM_Z, DIM_X, DIM_X);
            }

            // A = [((N_C-1) * DIM_Z), N_X]
            Mat::dot((float*)K, (float*)C, (float*)MH_a, DIM_X, DIM_Z, DIM_Z);

            Mat::pinv(
                    (float*) A, (float*) A_i,
                    (float*) ST, (float*) T_NN, (float*) T_MN,
                    (float*) U, (float*) W, (float*) V, (double*) rv1,
                    (DIM_C-1)*DIM_Z, DIM_X, 1e-9);

            Mat::dot((float*)A_i, (float*)(C + DIM_Z*DIM_Z), (float*)MH, DIM_X, DIM_Z, (DIM_C-1)*DIM_Z);
            Mat::add((float*)MH,(float*)MH_a,(float*)MH,DIM_X*DIM_Z);
            Mat::dot((float*)H, (float*)MH, (float*)R, DIM_Z, DIM_Z, DIM_X); //R = H*MH
            Mat::sub((float*)C, (float*)R, (float*)R, DIM_Z*DIM_Z); // R = C0 - H*MH
            Mat::abs((float*)R, (float*)R, DIM_Z*DIM_Z);
        }
};

#endif
