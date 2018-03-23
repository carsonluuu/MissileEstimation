#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 20:04:51 2018

@author: carsonluuu
"""

import numpy as np
import matplotlib.pyplot as plt
import random



"""

"""

def MonteCarlo(num, error_store, error_all, residual_all):
    
    Vc  = 300
    tf  = 10
    R1  = 15*(10**(-6))
    R2  = 1.67*(10**(-3))
    tau = 2
    W   = 100**2
    dt  = 0.01
    
    a_var  = 100.0**2
#    y_var  = 0
    v_var  = 200.0**2
    
    t = np.arange(0, 10, 0.01)
    
    length = len(t)
    timeList = range(0, length - 1)
    
    for index in range(num):
        if index%100 == 0:
            print(str(index/10) + "% has been done.")
        x_pre     = np.zeros((3, 1, length))
        dx_pre    = np.zeros((3, 1, length))
        xhat_pre  = np.zeros((3, 1, length))
        dxhat_pre = np.zeros((3, 1, length))
        
        
        F  = np.array([[0, 1, 0],
                      [0, 0, -1],
                      [0, 0.0, -(1/tau)]])
        G  = np.array([[0], [0], [1]])
#        B  = np.array([[0], [1], [0]])
        P0 = np.array([[0,   0,    0],
                       [0, 200**2, 0],
                       [0, 0, 100**2]])
        
        P_pre = np.zeros((3, 3, length))
        K_pre = np.zeros((3, 1, length))
        Z_pre = np.zeros((length))
        
        
        H0 = np.array([[1.0/(Vc*tf), 0, 0]])
        V0 = R1 + (R2/(tf**2))
        y0 = 0
        v0  = np.random.normal(0, v_var**0.5)
        
        at0 = (1 - 2 * round(random.uniform(0, 1))) * 100
        
        wat = np.random.normal(0, (a_var/dt)**0.5)
        n_var = V0/dt
        n = np.random.normal(0, n_var**0.5)
    
        error_pre    = np.zeros((3, 1, length))
        residual     = np.zeros((1, length))
        
        P_pre[:,:,0] = P0
        K_pre[:,:,0] = np.dot( np.dot(P0, H0.T), 
                                 V0**-1 )
        
        x_pre[:,:,0]    = np.array([[y0], [v0], [at0]])
        xhat_pre[:,:,0] = 0
        
        dx_pre[:,:,0]    = np.dot(F, x_pre[:,:,0]) + np.dot(G, wat)
        dxhat_pre[:,:,0] = np.dot(F, xhat_pre[:, :, 0]) \
                                    + np.dot(K_pre[:, :, 0], (Z_pre[0] \
                                    - np.dot(H0, xhat_pre[:, :, 0]))) 
    
        Z_pre[0] = np.dot(H0, x_pre[:,:,0]) + n
        error_pre[:,:,0] = 0
        
        t_x = -4*np.log(random.uniform(0, 1))
            
        for i in timeList:
            H = np.array([[1.0/(Vc*(tf-t[i])), 0.0, 0.0]])
            V = R1 + (R2/((tf-t[i])**2))
            P_dot = np.dot(F, P_pre[:,:,i]) \
                    + np.dot(P_pre[:,:,i], F.T) \
                    - np.dot(np.dot(np.dot(np.dot(P_pre[:,:,i], H.T), V**(-1)), H), P_pre[:,:,i]) \
                    + np.dot(np.dot(G, W), G.T)
            
            P_pre[:,:,i+1] = P_pre[:,:,i] + np.dot(P_dot, dt)
            K_pre[:,:,i+1] = np.dot(np.dot(P_pre[:,:,i], H.T), (V**(-1)))
            
            
            n = np.random.normal(0, (V/dt)**0.5)
            wat = np.random.normal(0, (a_var/dt)**0.5)
            
            if (t_x <= i*dt) :
                at0 = -at0
                x_pre[2,0,i] = at0
                t_x += -4*np.log(random.uniform(0, 1))
            else:
                x_pre[2,0,i] = at0
            
            dx_pre[:,:,i+1] = np.dot(F, x_pre[:,:,i]) + np.dot(G, wat)
            x_pre[:,:,i+1]  = x_pre[:,:,i] + dx_pre[:,:,i+1]*dt
            Z_pre[i+1]      = np.dot(H, x_pre[:,:,i+1]) + n
            
            dxhat_pre[:,:,i+1] = np.dot(F, xhat_pre[:,:,i]) +\
                                 np.dot(K_pre[:,:,i+1] , (Z_pre[i+1] - np.dot(H, xhat_pre[:,:,i])))
            xhat_pre[:,:,i+1]  = xhat_pre[:,:,i] + dxhat_pre[:,:,i+1]*dt
    
            residual[:,i+1]            = Z_pre[i+1] - np.dot(H, xhat_pre[:,:,i+1])
            residual_all[:,i+1, index] = residual[:,i+1]
            error_pre[:,:,i+1]         = xhat_pre[:,:,i+1] - x_pre[:,:,i+1]
            error_store[:,:,i+1,index] = error_pre[:,:,i+1]
        
        error_all += error_store[:,:,:,index]
    print("Monte Carlo simulation has finished with " + str(num) + " realizations")
    return K_pre, P_pre, x_pre, xhat_pre, error_all, error_store, error_pre, residual_all

def avg(P_out, error, error_store, residual_all, num):
    
    error_avg = error/num
    res_chk = 0
    
    for i in range(0, num):
        res_chk = res_chk + np.dot(residual_all[:,40,i], residual_all[:,80,i].T)
        for j in range(0, length):
            P_out[:,:,j] += np.dot((error_store[:,:,j,i] - \
                    error_avg[:,:,j]), (error_store[:,:,j,i] - error_avg[:,:,j]).T)
    
    _res_chk = res_chk/num
    res_chk = _res_chk
    print(res_chk)
    return P_out
    
def plot_Gain(K):    
    
    plt.figure(1)
    plt.title('Filter Gain History')
    plt.plot(t, np.squeeze(K[0,:,:].T),label = "K1")
    plt.plot(t, np.squeeze(K[1,:,:].T), linestyle='--', label = "K2")
    plt.plot(t, np.squeeze(K[2,:,:].T), linestyle=':', label = "K3")
    plt.ylabel('Kalman Filter Gains')
    plt.xlabel('Time After Launching [s]')
    plt.legend(loc="best")

def plot_RMS(P):
    
    plt.figure(2)
    plt.title('Evolution of the Estimation Error RMS')
    plt.plot(t, np.squeeze(P[0, 0, :]**0.5), label = "RMS for Position")
    plt.plot(t, np.squeeze(P[1, 1, :]**0.5), linestyle='--', label = "RMS for Velocity")
    plt.plot(t, np.squeeze(P[2, 2, :]**0.5), linestyle=':', label = "RMS for Acceleration")
    plt.ylabel('Standard Deviation of the State Error')
    plt.xlabel('Time After Launching [s]')
    plt.legend(loc="best")

    
def plot_Act_Est(xhat, x): 
    
    plt.figure(3, figsize=(6, 2))
    plt.title('Actual vs. Estimate for Position')
    plt.plot(t, np.squeeze(xhat[0, 0, :]), label = "Actual")
    plt.plot(t, np.squeeze(x[0, 0, :]), linestyle='--', label = "Estimated")
    plt.ylabel('Position')
    plt.xlabel('Time After Launching [s]')
    plt.legend(loc="best")

    plt.figure(4, figsize=(6, 2))
    plt.title('Actual vs. Estimate for Velocity')
    plt.plot(t, np.squeeze(xhat[1, 0, :]), label = "Actual")
    plt.plot(t, np.squeeze(x[1, 0, :]), linestyle='--', label = "Estimated")
    plt.ylabel('Velocity')
    plt.xlabel('Time After Launching [s]')
    plt.legend(loc="best")

    plt.figure(5, figsize=(6, 2))
    plt.title('Actual vs. Estimate for Acceleration')
    plt.plot(t, np.squeeze(xhat[2, 0, :]), label = "Actual")
    plt.plot(t, np.squeeze(x[2, 0, :]), linestyle='--', label = "Estimated")
    plt.ylabel('Acceleration')
    plt.xlabel('Time After Launching [s]')
    plt.legend(loc="best")
    
def plot_Error_Variance(P_out_avg, P):
    
    plt.figure(6)
    plt.title('Actual Error Variance vs. a priori Error Variance for Position')
    plt.plot(t, P_out_avg[0, 0, :], label = "a priori Error")
    plt.plot(t, P[0, 0, :], linestyle='--', label = "Actual")
    plt.ylabel('Position')
    plt.xlabel('Time After Launching [s]')
    plt.legend(loc="best")
    
    plt.figure(7)
    plt.title('Actual Error Variance vs. a priori Error Variance for Velocity')
    plt.plot(t, P_out_avg[1, 1, :], label = "a priori Error")
    plt.plot(t, P[1, 1, :], linestyle='--', label = "Actual")
    plt.ylabel('Velocity')
    plt.xlabel('Time After Launching [s]')
    plt.legend(loc="best")
    
    plt.figure(8)
    plt.title('Actual Error Variance vs. a priori Error Variance for Acceleration')
    plt.plot(t, P_out_avg[2, 2, :], label = "a priori Error")
    plt.plot(t, P[2, 2, :], linestyle='--', label = "Actual")
    plt.ylabel('Acceleration')
    plt.xlabel('Time After Launching [s]')
    plt.legend(loc="best")
    
def plot_Error(error, P):
    
    plt.figure(9, figsize=(9, 3))
    plt.title('Position Error for One Single Run')
    plt.plot(t, np.squeeze(error[0,:,:].T), label = "Position Error")
    plt.plot(t, P[0, 0, :]**0.5, linestyle='--', label = r'$\sigma$')
    plt.plot(t, -P[0, 0, :]**0.5, linestyle='--', label = r'$-\sigma$')
    plt.ylabel('Position')
    plt.xlabel('Time After Launching [s]')
    plt.legend(loc="best")
    
    plt.figure(10, figsize=(9, 3))
    plt.title('Velocity Error for One Single Run')
    plt.plot(t, np.squeeze(error[1,:,:].T), label = "Velocity Error")
    plt.plot(t, P[1, 1, :]**0.5, linestyle='--', label = r'$\sigma$')
    plt.plot(t, -P[1, 1, :]**0.5, linestyle='--', label = r'$-\sigma$')
    plt.ylabel('Velocity')
    plt.xlabel('Time After Launching [s]')
    plt.legend(loc="best")
    
    plt.figure(11, figsize=(9, 3))
    plt.title('Acceleration Error for One Single Run')
    plt.plot(t, np.squeeze(error[2,:,:].T), label = "Acceleration Error")
    plt.plot(t, P[2, 2, :]**0.5, linestyle='--', label = r'$\sigma$')
    plt.plot(t, -P[2, 2, :]**0.5, linestyle='--', label = r'$-\sigma$')
    plt.ylabel('Acceleration')
    plt.xlabel('Time After Launching [s]')
    plt.legend(loc="best")


if __name__ == "__main__":
    
    num = 1
    t = np.arange(0, 10, 0.01)      
    length = len(t)
            
    P_out        = np.zeros((3, 3, length))
    error_store  = np.zeros((3, 1, length, num))
    error_all    = np.zeros((3, 1, length))
    residual_all = np.zeros((1, length, num))

    K, P, x, xhat, error, error_store, error_, residual_all \
        = MonteCarlo(num, error_store, error_all, residual_all)
    P_out_avg = avg(P_out, error, error_store, residual_all, num)
    
#    plot_Gain(K)
#    plot_RMS(P)
    plot_Act_Est(xhat, x)
#    plot_Error_Variance(P_out_avg/(num-1), P)
#    plot_Error(error_, P) #single
#    plot_Error(error/100, P) #average
    
    