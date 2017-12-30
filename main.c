#include <cstring>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <vector>

#include <random>

#include "mat_ops.h"
#include "kalman.h"

#define N_X 4
#define N_Z 2

#define N 100
#define N_L 50 //lag
#define N_C (N_L+1)

#define DT 0.1

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

void autocorrelation(
        float* Y, float* T_ZZ, float* C,
        int n_y, int n_c){
    // Y = [N, N_Z]
    memset(C, 0, n_c*N_Z*N_Z*sizeof(float));
    for(int k=0; k<n_c; ++k){
        float* Ck = C+k*N_Z*N_Z;
        for(int i=k; i<n_y; ++i){
            Mat::dot(Y+i*N_Z, Y+(i-k)*N_Z, T_ZZ, N_Z, N_Z, 1);
            Mat::add(Ck, T_ZZ, Ck, N_Z*N_Z);
        }
        // TODO : divide by (N_Y-k+1) instead of N_Y??
        Mat::div(Ck, n_y, Ck, N_Z*N_Z);
    }
}

#define T_N 3
#define T_M 2
int main(){
    srand(time(0));

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> d(0, sqrt(0.01));

    //float TSRC[T_N][T_M] = {
    //    {0.418, 0.287},
    //    {0.513, 0.59},
    //    {0.202, 0.948}
    //};
    //float TDST[T_M][T_N]={};
    //float T_NN[T_N][T_N]={};
    //float T_MN[T_M][T_N]={};
    //float u[T_N][T_N]={};
    //float v[T_M][T_M]={};
    //float ST[T_M][T_N]={};

    //float eps = 1e-6;
    //double rv1[T_N>T_M?T_N:T_M]={};
    //float w[T_N<T_M?T_M:T_N]={};
    //for(int i=0; i<T_N; ++i){
    //    for(int j=0; j<T_M; ++j){
    //        u[i][j] = TSRC[i][j];
    //    }
    //}
    //Mat::pinv((float*)TSRC, (float*)TDST, 
    //        (float*)ST, (float*)T_NN, (float*)T_MN,
    //        (float*)u, (float*)w, (float*)v, (double*)rv1,
    //        T_N, T_M, eps);
    //Mat::print((float*)TDST, T_M, T_N);
    
    float x[N_X][1] = {
        {1},
        {0},
        {0},
        {0},
    }; // state
    float P[N_X][N_X] = {
        {0,0,0,0},
        {0,0,0,0},
        {0,0,100,0},
        {0,0,0,100}
    }; // covariance

    float DT4 = 0.25 * DT*DT*DT*DT;
    float DT3 = 0.5 * DT*DT*DT;
    float DT2 = DT*DT;

    //float Q[N_X][N_X] = {
    //    {DT4,0,DT3,0},
    //    {0,DT4,0,DT3},
    //    {DT3,0,DT2,0},
    //    {0,DT3,0,DT2}
    //}; // process noise
    //Mat::mul((float*)Q, 1*1, (float*)Q, N_X*N_X);
    float Q[N_X][N_X] = {
        {0.006,-0.0005,9.46e-5,-7.44e-5},
        {-0.0005,0.0009,-5e-5,0.00015},
        {9.46e-5,-5e-5,1.36e-5,-7.67e-6},
        {-7.44e-5,0.00015,-7.67e-6,2.65e-5}
    }; // process noise
    float R[N_Z][N_Z] = {
        {0.007,0.0005},
        {0.0005,0.007}
    }; // measurement noise - est
    float F[N_X][N_X] = {
        {1,0,DT,0},
        {0,1,0,DT},
        {0,0,1,0},
        {0,0,0,1}
    }; //state transition
    float H[N_Z][N_X] = {
        {1,0,0,0},
        {0,1,0,0}
    }; // observation mapping

    float z[N][N_Z];
    float Y[N][N_Z];

    std::vector<float> 
        obs_x, obs_y, 
        est_x, est_y,
        err_x, err_y,
        gt_x, gt_y;


    obs_x.reserve(N);
    obs_y.reserve(N);

    est_x.reserve(N);
    est_y.reserve(N);

    gt_x.reserve(N);
    gt_y.reserve(N);

    KalmanFilter<N_X,N_Z,N_C> KF((float*)P,(float*)Q,(float*)R,(float*)F,(float*)H,(float*)x,true);
    for(int i=0; i<N; ++i){
        float t = i*DT*M_PI/(N*DT);
        float x = 2*cos(t) - 2*sin(t);
        float y = sin(t) + cos(t);
        //float y = 0;//2*x;

        gt_x.push_back(x);
        gt_y.push_back(y);

        z[i][0] = x + d(gen);
        z[i][1] = y + d(gen);

        KF.predict();
        KF.update(z[i]);

        obs_x.push_back(z[i][0]);
        obs_y.push_back(z[i][1]);

        est_x.push_back(KF.x[0]);
        est_y.push_back(KF.x[1]);

        err_x.push_back(KF.y[0]);
        err_y.push_back(KF.y[1]); // residuals

        Y[i][0] = KF.y[0];
        Y[i][1] = KF.y[1]; //residuals

        plt::clf();
        plt::plot(obs_x, obs_y);
        plt::plot(est_x, est_y);
        plt::plot(gt_x, gt_y);
        //plt::xlim(-1.5, 1.5);
        //plt::ylim(-1.5, 1.5);
        plt::pause(0.001);
    }
    plt::show();

    Mat::print((float*) KF.Q, N_X, N_X);
    std::cout << "++++++++++++++" << std::endl;
    //plt::plot(obs_x, obs_y);
    //plt::plot(est_x, est_y);
    //plt::plot(gt_x, gt_y);
    //plt::show();

    float C[N_C][N_Z][N_X];
    float T_ZZ[N_Z][N_Z];
    autocorrelation((float*)Y, (float*)T_ZZ, (float*)C, N, N_C);

    // estimate R matrix
    Mat::print((float*) KF.R, N_Z, N_Z);
    KF.ALS_R((float*)C);
    Mat::print((float*) KF.R, N_Z, N_Z);

    return 0;
}
