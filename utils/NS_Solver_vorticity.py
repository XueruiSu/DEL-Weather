"""
@author: Xuerui Su
from FNO's NS Slover
"""

import torch
# from utils.utilities3 import *
import math

from data_generation.navier_stokes.random_fields import GaussianRF

from timeit import default_timer

import scipy.io

import mat73

#w0: initial vorticity (B,H,W)
#f: forcing term
#visc: viscosity (1/Re)
#T: final time
#delta_t: internal time-step for solve (descrease if blow-up)
#record_steps: number of in-time snapshots to record
def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):

    #Grid size - must be power of 2
    N = w0.size()[-1]

    #Maximum frequency
    k_max = math.floor(N/2.0)

    #Number of steps to final time
    steps = math.ceil(T/delta_t)

    #Initial vorticity to Fourier space
    w_h = torch.fft.rfft2(w0)

    #Forcing to Fourier space
    f_h = torch.fft.rfft2(f)

    #If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    #Record solution every this number of steps
    record_time = math.floor(steps/record_steps)

    #Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device), torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N,1)
    #Wavenumbers in x-direction
    k_x = k_y.transpose(0,1)

    #Truncate redundant modes
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]

    #Negative Laplacian in Fourier space
    lap = 4*(math.pi**2)*(k_x**2 + k_y**2)
    lap[0,0] = 1.0
    #Dealiasing mask
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0/3.0)*k_max, torch.abs(k_x) <= (2.0/3.0)*k_max).float(), 0)

    #Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    #Record counter
    c = 0
    #Physical time
    t = 0.0
    for j in range(steps):
        #Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / lap

        #Velocity field in x-direction = psi_y
        q = 2. * math.pi * k_y * 1j * psi_h
        q = torch.fft.irfft2(q, s=(N, N))

        #Velocity field in y-direction = -psi_x
        v = -2. * math.pi * k_x * 1j * psi_h
        v = torch.fft.irfft2(v, s=(N, N))

        #Partial x of vorticity
        w_x = 2. * math.pi * k_x * 1j * w_h
        w_x = torch.fft.irfft2(w_x, s=(N, N))

        #Partial y of vorticity
        w_y = 2. * math.pi * k_y * 1j * w_h
        w_y = torch.fft.irfft2(w_y, s=(N, N))

        #Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.rfft2(q*w_x + v*w_y)

        #Dealias
        F_h = dealias* F_h

        #Crank-Nicolson update
        w_h = (-delta_t*F_h + delta_t*f_h + (1.0 - 0.5*delta_t*visc*lap)*w_h)/(1.0 + 0.5*delta_t*visc*lap)

        #Update real time (used only for recording)
        t += delta_t
        
        if (j+1) % record_time == 0:
            #Solution in physical space
            w = torch.fft.irfft2(w_h, s=(N, N))

            #Record solution and time
            sol[...,c] = w
            sol_t[c] = t

            c += 1


    return sol, sol_t


def solver_ns(ut, batch_size_first=True, visc=1e-3, Solver_T=1, delta_t=5e-3):
    # change ut to tensor and permute to (B,H,W)
    if isinstance(ut, torch.Tensor):
        if batch_size_first:
            pass
        else:
            ut = ut.permute(2,0,1)
    else:
        if batch_size_first:
            ut = torch.tensor(ut)
        else:
            ut = torch.tensor(ut).permute(2,0,1)
                
    # Resolution
    s = 64

    # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    t = torch.linspace(0, 1, s+1)
    t = t[0:-1]

    # grid
    X,Y = torch.meshgrid(t, t, indexing='ij')
    f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))
    f = f.to(device=ut.device)
    
    sol, sol_t = navier_stokes_2d(ut, f, visc, Solver_T, delta_t=delta_t, record_steps=Solver_T)
    return sol


