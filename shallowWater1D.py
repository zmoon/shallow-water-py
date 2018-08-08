# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:46:29 2017

This is v0.2


---------------------- History ---------------------------------------------------------------------

*...v0.1:
modified from Drew's initial version
added explicit advancing of the analytical adv-only soln

*...v0.2: 
Schemes now use spatial index vectors from model class attribute 
  that change based on whether cyclic BC is set or not
  don't need ghost points anymore for cyclic
Schemes also use local vars in the calculations to make the math easier to read
  and diffusion added to numerical scheme methods, 
  since same scheme should be used as in the rest of the math (? check this with Stauffer)
RK methods added, though not totally sure if implementing correctly
  and, the indexing in the RK schemes needs to be fixed for non-cyclic BC to work
should also add:
  - calculation of phase speed in numerical solution, for every step
  - higher order diffusion (i.e., del-4, del-6 for the CT schemes)
  - CTCS6, RK3CS6
  - u and v for RK methods, and fix indexing so non-cyclic BCs work correctly
could have separated time and space parts of the schemes instead of double-coding the CS's (in CT and RK)

----------------------------------------------------------------------------------------------------

@author: zmoon
"""

import matplotlib.pyplot as plt
#import numba
import numpy as np
from scipy import interpolate


# could also have a helper fn to get the inds for cyclic vs non-cyclic, 
# then wouldn't have to put them as arguments in all of the helper FD fns
# or put as methods in the class!
# but could also use numba jit here for speed? not sure if it would work in class
# will have to test.


## ---- 1st derivative
#
#def CS2d1(y, im1, i, ip1):
#    """ centered-in-space FD 1st deriv
#    
#    -1/2  (0)  1/2
#    """
#    return 0.5*(y[ip1] - y[im1])
#
#
#def CS4d1(y, im2, im1, i, ip1, ip2):
#    """ centered-in-space FD 1st deriv
#    
#    1/12  −2/3  (0)  2/3  −1/12 	
#    """
#    return 1.0/12*(-y[ip2] + 8*y[ip1] + -8*y[im1] + y[im2])
#
#
#def CS6d1(y, im3, im2, im1, i, ip1, ip2, ip3):
#    """ centered-in-space FD 1st deriv
#    
#    −1/60  3/20  −3/4 (0)  3/4  −3/20  1/60 	
#    """
#    return 1.0/60*(y[ip3] + -9*y[ip2] + 45*y[ip1] + -45*y[im1] + 9*y[im2] + -y[im3])
#
#
## ---- 2nd derivative
#
#def CS2d2(y, im1, i, ip1):
#    """ centered-in-space FD 2nd deriv
#    
#    1  -2  1
#    """
#    return y[im1] - 2*y[i] + y[ip1]
#    
#
#def CS4d2(y, im2, im1, i, ip1, ip2):
#    """ centered-in-space FD 2nd deriv
#    for y(xg) where xg points are equally spaced
#    
#    −1/12  4/3  −5/2  4/3  −1/12
#    """
#    return 1.0/12 * (-1*y[im2] + 16*y[im1] + -30*y[i] + 16*y[ip1] + -1*y[ip2])
#
#
#def CS6d2(y, im3, im2, im1, i, ip1, ip2, ip3):
#    """ centered-in-space FD 2nd deriv
#    
#    1/90  −3/20  3/2  −49/18  3/2  −3/20  1/90 	
#    """
#    return 1.0/180 * (2*y[im3] + -27*y[im2] + 270*y[im1] + -490*y[i] + \
#                      270*y[ip1] + -27*y[ip2] + 2*y[ip3])
#
#
## ---- 4th derivative
#    
#def CS2d4(y, im2, im1, i, ip1, ip2)
#    """ centered-in-space FD 4th deriv
#    
#    1  -4  6  -4  1
#    """
#    return 1*y[im2] + -4*y[im1] + 6*y[i] + -4*y[ip1] + 1*y[ip2]
#
#
#def CS4d4(y, im3, im2, im1, i, ip1, ip2, ip3)
#    """ centered-in-space FD 4th deriv
#    
#    −1/6  2  −13/2  28/3  −13/2  2  −1/6
#    """
#    return 1.0/6*(-1*y[im3] + 12*y[im2] + -78*y[im1] + 168*y[i] + -78*y[ip1] + 12*y[ip2] + -1*y[ip3])



class model():
    """ for 1-D shallow water linear advection problem 
    
    linearized about state of rest:
      v_bar = 0
      u_bar = 0
      h_bar = const
      h = h_bar + h', etc
      
    wind only in x-dir:
      v' = 0
      u' = C
      
    homogeneous in y, remove primes for clarity, gives the following system:
                 adv      div?     Co     diffusion
      dh/dt = -u*dh/dx - h*du/dx            (+ Dh)
      du/dt = -u*du/dx            + f*v     (+ Du)     
      dv/dt = -u*dv/dx            - f*u     (+ Dv)
    where D is an explicit diffusion term formed from the Laplacian (diff2)
    or higher order derivatives (diff4, diff6)

    
    """
    
    def __init__(self, 
                 dx=1e5, dt=500, Nx=50, Nt=200,
                 C=10.0, hbar=8000.0, amp=100.0, hc=25, hw=9,
                 g=9.8, f=1e-4, kdif=None, 
                 nifcor=0, nifwnd=0, nifdif=0, nifadv=1,
                 adv_scheme='CTCS2', bc='cyclic', 
                 time_filter_scheme=None, time_filter_beta=None, time_filter_alpha=None, 
                 show_anim=True, anim_fps=5, anim_runtime=10, anim_stride=None, 
                 show_analytic=True, anim_pause_on=[],
                 ctcs_dif_scheme='CS2d2',
                 ):
        """
        numerics parameters                                   default
           dx: horiz grid spacing (m)                           1e5
           dt: time step (s)                                    500
           nx: number of grid pts (including 2 ghost nodes)     50 (was 50 + 2 ghost points in the class code)
           nt: number of time steps                             200
        
        initial condition                                     default
           C: advection velocity (m/s)                          10
           hbar: base level of height (m)                       8000
           amp: amplitude of the height wave perturbation       100
           hc: center location of initial perturbation          25 (in dx)
           hw: width of hill (in dx)                            9 (even numbers round down to odd)
        
        physics parameters
           g: accel. due to gravity (m/s)
           f: Coriolis param
           kdif: coeff that multiplies the diffusion term
        
        forcing term switches
           nifcor: Coriolis
           nifwind: changes in wind over time
           nifdif: explicit diffusion
           nifadv: advection
           
        numerical scheme
           adv_scheme: {'CTCS{2,4}', 'FTCS2', FTBS', 'RK{3,4}CS{4,6}'}
           bc: {'cyclic', 'open'}
           time_filter_alpha: [0, 1], with 0.25 giving a 1-2-1 time smoother
           
        animation settings
           show_anim: T/F
           anim_fps: desired number of frames per sec (neglecting integration computation time)
           anim_runtime: desired animation run time (s)
           ...
           
        
        """

        # numerics parameters
        self.dx = float(dx)
        self.dt = float(dt)
        self.Nx = int(Nx)
        self.Nt = int(Nt)
        
        # initial condition parameters
        self.C    = C
        self.hbar = hbar
        self.amp  = amp
        self.hc   = hc
        self.hw   = hw
        self.blowup_cutoff = (hbar + amp) * 1e3
        
        # physics parameters
        self.g = g
        self.f = f
        self.kdif = kdif or 0.01 * self.dx**2 / (2*self.dt)  # could also use 'if not else' syntax
        
        # forcing term switches
        self.nifcor  = nifcor  # Coriolis
        self.nifwnd = nifwnd  # changes in wind over time
        self.nifdif  = nifdif  # explicit diffusion
        self.nifadv  = nifadv  # advection
        
        self.ctcs_dif_scheme = ctcs_dif_scheme
        self.time_filter_scheme = time_filter_scheme  # {'Asselin', 'RAW'} also Lax-Wendroff, though that should really be somewhere else
        self.time_filter_beta  = time_filter_beta  # =0.5 is a 1-2-1 time smoother for Asselin
        self.time_filter_alpha = time_filter_alpha  # used in RAW (Robert-Asselin-William)
        
        # numerical scheme
        self.adv_schemes = dict(CTCS2=self.CTCS2, CTCS4=self.CTCS4, CTCS6=self.CTCS6,
                                FTCS2=self.FTCS2, 
                                FTBS=self.FTBS,
                                RK3CS4=self.RK3CS4, RK4CS4=self.RK4CS4, RK4CS6=self.RK4CS6,
                                spectral=self.spectral, spectralInterp=self.spectralInterp,
                                )
        self.adv_scheme = self.adv_schemes[adv_scheme]
        self.adv_scheme_name = adv_scheme
        self.bc = bc
        if self.bc == 'cyclic':  # could do this stuff a smarter way by creating just one vector that has all the inds, but would have to change the schemes
            self.i = np.arange(Nx)
            self.ip3 = np.hstack((self.i[3:], self.i[:3]))
            self.ip2 = np.hstack((self.i[2:], self.i[:2]))
            self.ip1 = np.hstack((self.i[1:], self.i[:1]))
            self.im1 = np.hstack((self.i[-1:], self.i[:-1]))
            self.im2 = np.hstack((self.i[-2:], self.i[:-2]))
            self.im3 = np.hstack((self.i[-3:], self.i[:-3]))
        elif self.bc == 'open':  # or could modify the cyclic versions by setting outer ones to nan?
            CS_order = self.adv_scheme_name[self.adv_scheme_name.index('CS')+2]
            if CS_order == '2':
                self.i = np.arange(1, Nx-1, 1)
                self.ip1 = self.i + 1
                self.im1 = self.i - 1
            elif CS_order == '4':
                self.i = np.arange(2, Nx-2, 1)
                self.ip2 = self.i + 2
                self.ip1 = self.i + 1
                self.im1 = self.i - 1
                self.im2 = self.i - 2
            elif CS_order == '6':
                self.i = np.arange(3, Nx-3, 1)
                self.ip3 = self.i + 3
                self.ip2 = self.i + 2
                self.ip1 = self.i + 1
                self.im1 = self.i - 1
                self.im2 = self.i - 2
                self.im3 = self.i - 3
        else:
            pass
        
        # animation settings
        self.show_anim = show_anim
        self.anim_fps = anim_fps
        self.anim_runtime = anim_runtime
        self.anim_stride = anim_stride or np.floor(self.Nt/(self.anim_runtime*self.anim_fps)) or 1
        self.anim_num_frames = int(np.floor(self.Nt / self.anim_stride))
        self.i_frame = 0
        self.anim_delay = 1.0/self.anim_fps
        self.show_analytic = show_analytic
#        self.show_wind = False
#        self.wind_scale_factor = self.Nx
        self.anim_pause_on = anim_pause_on  # pause on certain steps
        
        # initialize the vars
        self.initialize()
        
        
    def initialize(self):
        """ """
        
        self.x = np.arange(self.Nx)*self.dx  # the model grid (m)

        self.u = np.full((self.Nx,), self.C, dtype=np.float64)
        self.v = np.full((self.Nx,), 0, dtype=np.float64)
        self.h = np.full((self.Nx,), self.hbar, dtype=np.float64)
        
        initial_pertub_i = np.arange(np.ceil(self.hc-self.hw/2.0), np.ceil(self.hc+self.hw/2.0), dtype=int) 
        self.h[initial_pertub_i] = self.hbar + self.amp/2.0*(1 + np.cos(2*np.pi * (initial_pertub_i-self.hc)/self.hw) )
        
        # some schemes need this defined at initial time step (and so does time filter)
        self.hb = self.h
        self.ub = self.u
        self.vb = self.v
        
        # store initial state
        self.h0 = np.copy(self.h)  # store initial values of h, u, v
        self.u0 = np.copy(self.u)
        self.v0 = np.copy(self.v)
        self.x0 = np.copy(self.x)  # for analytical solution
        self.x00 = np.copy(self.x0)
        self.i_hmax0 = np.argmax(self.h)
        self.hmax0 = self.hbar + self.amp
        
        # pre-allocate 
        self.i_hmax = np.zeros((self.Nt+1,))
        self.hmax = np.zeros_like(self.i_hmax)
        self.i_hmax[0] = self.i_hmax0
        self.hmax[0] = self.hmax0
        
        # save some stuff here
        self.data = dict(x=self.x, 
                         nt=np.zeros((self.anim_num_frames,), dtype=int),
                         h=np.zeros((self.anim_num_frames, self.Nx), dtype=np.float64),
                         u=np.zeros((self.anim_num_frames, self.Nx), dtype=np.float64),
                         v=np.zeros((self.anim_num_frames, self.Nx), dtype=np.float64),
                         x0=np.zeros((self.anim_num_frames, self.Nx), dtype=np.float64),  # x for analytical soln
                         h0=np.zeros((self.anim_num_frames, self.Nx), dtype=np.float64),  # h for analytical soln
                         u0=np.zeros((self.anim_num_frames, self.Nx), dtype=np.float64),
                         v0=np.zeros((self.anim_num_frames, self.Nx), dtype=np.float64),
                         )
        
        # create figs and axes
        if self.show_anim:
            self.fig1, self.fig1ax1 = plt.subplots(figsize=(5, 3))#, num='sim')


    def run(self):
        """ """
        
        # could check if already been run and give error
        # or call initialize() here instead
        
        for self.nt in range(1, self.Nt+1):  # maybe should start at 0 since I moved the 'advance time'
        
            # calculate values at the next timestep (nt)
            #   instead of returning values in the scheme methods, could just modify the self.f's
            self.hf, self.uf, self.vf = self.adv_scheme()
            
            # check if blow up
            if np.any(np.abs(self.hf) > self.blowup_cutoff):
                print('model blew up at time step {nt:d}'.format(nt=self.nt))
                break
            
            # apply time filter?
            if self.time_filter_beta:
                nu = self.time_filter_beta
                
                if self.time_filter_scheme == 'Asselin':  # basic time smoother
#                    print 'applying basic time filter...'
                    
                    self.h += nu/2*(self.hb-2*self.h+self.hf)
                
                elif self.time_filter_scheme == 'RAW':
                    alpha = self.time_filter_alpha
                    
#                    print 'applying RAW time filter...'
                    
                    d = nu/2*(self.hb-2*self.h+self.hf)
                    
                    self.h += d*alpha
                    self.hf += d*(alpha-1)
                    
                    
            
            # also find better way to find the x pos of the numerical soln peak
            #   perhaps average the 3 or 4 peaks with highest h?
            
            
            # plot results? and store data
            if self.nt % self.anim_stride == 0 or np.any(self.nt == self.anim_pause_on):
                
                if self.show_anim:
                    self.animate()
                
                # add to data dict
                self.data['h'][self.i_frame,:] = self.h
                self.data['u'][self.i_frame,:] = self.u
                self.data['v'][self.i_frame,:] = self.v
                self.data['x0'][self.i_frame,:] = self.x0
                self.data['h0'][self.i_frame,:] = self.h0
                self.data['nt'][self.i_frame] = self.nt - 1  # time not advanced yet
                self.i_frame += 1
                
            # advance time
            self.hb, self.h = self.h, self.hf  # current timestep becomes the backward, and forward timestep becomes the current
            self.ub, self.u = self.u, self.uf
            self.vb, self.v = self.v, self.vf
            
            # advance analytical solution position (simply moves forward with speed C)
            #   incorpating the cyclic BC if necessary
            if not self.nifwnd:
                self.x0 += self.C*self.dt
                if self.bc == 'cyclic':
                    beyond = self.x0 > self.x.max()
                    self.x0 = np.hstack((self.x0[beyond]-(self.Nx-0)*self.dx, self.x0[~beyond]))
                    self.h0 = np.hstack((self.h0[beyond], self.h0[~beyond])) 
                
                
        # write out saved values?
                
        
            
    def animate(self):
        """ """
        
        self.fig1ax1.cla()
        self.fig1ax1.plot(self.x/1000, self.h, '.-', lw=1, ms=4)
        
        if not self.nifwnd:
            self.fig1ax1.plot(self.x0/1000, self.h0, 'r:', lw=1.5)
        
        self.fig1ax1.set_xlim((self.x.min()/1000, self.x.max()/1000))
        self.fig1ax1.set_ylim((self.hbar-0.5*self.amp, self.hbar+1.2*self.amp))
        self.fig1ax1.set_xlabel('x (m)')
        self.fig1ax1.set_ylabel('h (m)')
        s = 'n = {nt:0{padnt:}d}, t = {t:0{padt:}.2f} h'.format(nt=self.nt, t=self.nt*self.dt/3600, 
                 padt='{:d}'.format(len(str(np.ceil(self.Nt*self.dt/3600)))+1), padnt='{:d}'.format(len(str(self.Nt))))
        self.fig1ax1.set_title(s)
        self.fig1.tight_layout()        
        
        if np.any(self.nt == self.anim_pause_on):
            plt.waitforbuttonpress()
            plt.draw()
            
        else:
            plt.pause(self.anim_delay)
            

    # --- 1st derivative
    
    def CS2d1(self, y):
        """ centered-in-space FD 1st deriv
        
        -1/2  (0)  1/2
        """
        ip1 = self.ip1
        im1 = self.im1
        
        return 0.5*(y[ip1] - y[im1])
    
    
    def CS4d1(self, y):
        """ centered-in-space FD 1st deriv
        
        1/12  −2/3  (0)  2/3  −1/12 	
        """
        ip2 = self.ip2
        ip1 = self.ip1
        im1 = self.im1
        im2 = self.im2
        
        return 1.0/12*(-y[ip2] + 8*y[ip1] + -8*y[im1] + y[im2])
    
    
    def CS6d1(self, y):
        """ centered-in-space FD 1st deriv
        
        −1/60  3/20  −3/4 (0)  3/4  −3/20  1/60 	
        """
        ip3 = self.ip3
        ip2 = self.ip2
        ip1 = self.ip1
        im1 = self.im1
        im2 = self.im2
        im3 = self.im3
        
        return 1.0/60*(y[ip3] + -9*y[ip2] + 45*y[ip1] + -45*y[im1] + 9*y[im2] + -y[im3])
    
    
    # --- 2nd derivative
    
    def CS2d2(self, y):
        """ centered-in-space FD 2nd deriv
        
        1  -2  1
        """
        ip1 = self.ip1
        i   = self.i
        im1 = self.im1
        
        return y[im1] - 2*y[i] + y[ip1]
        
    
    def CS4d2(self, y):
        """ centered-in-space FD 2nd deriv
        for y(xg) where xg points are equally spaced
        
        −1/12  4/3  −5/2  4/3  −1/12
        """
        ip2 = self.ip2
        ip1 = self.ip1
        i   = self.i
        im1 = self.im1
        im2 = self.im2
        
        return 1.0/12 * (-1*y[im2] + 16*y[im1] + -30*y[i] + 16*y[ip1] + -1*y[ip2])
    
    
    def CS6d2(self, y):
        """ centered-in-space FD 2nd deriv
        
        1/90  −3/20  3/2  −49/18  3/2  −3/20  1/90 	
        """
        ip3 = self.ip3
        ip2 = self.ip2
        ip1 = self.ip1
        i   = self.i
        im1 = self.im1
        im2 = self.im2
        im3 = self.im3
        
        return 1.0/180 * (2*y[im3] + -27*y[im2] + 270*y[im1] + -490*y[i] + \
                          270*y[ip1] + -27*y[ip2] + 2*y[ip3])
    
    
    # --- 4th derivative
        
    def CS2d4(self, y):
        """ centered-in-space FD 4th deriv
        
        1  -4  6  -4  1
        """
        ip2 = self.ip2
        ip1 = self.ip1
        i   = self.i
        im1 = self.im1
        im2 = self.im2
        
        return 1*y[im2] + -4*y[im1] + 6*y[i] + -4*y[ip1] + 1*y[ip2]
    
    
    def CS4d4(self, y):
        """ centered-in-space FD 4th deriv
        
        −1/6  2  −13/2  28/3  −13/2  2  −1/6
        """
        ip3 = self.ip3
        ip2 = self.ip2
        ip1 = self.ip1
        i   = self.i
        im1 = self.im1
        im2 = self.im2
        im3 = self.im3
        
        return 1.0/6*(-1*y[im3] + 12*y[im2] + -78*y[im1] + 168*y[i] + -78*y[ip1] + 12*y[ip2] + -1*y[ip3])
    
    
    # --- 6th derivative

    def CS2d6(self, y):
        """ centered-in-space FD 6th deriv
        
        """
        ip3 = self.ip3
        ip2 = self.ip2
        ip1 = self.ip1
        i   = self.i
        im1 = self.im1
        im2 = self.im2
        im3 = self.im3
        
        return 1.0*(1*y[im3] + -6*y[im2] + 15*y[im1] + -20*y[i] + 15*y[ip1] + -6*y[ip2] + 1*y[ip3])


    # --- numerical schemes 

#    def CTCSx(self):
#        """ 2nd-O CT + whatever CS order is set (2, 4, 6)
#        aka leap-frog 
#        """
#        
#        # pre-allocate forward soln
#        hf = np.copy(self.hb)
#        uf = np.copy(self.ub)
#        vf = np.copy(self.vb)
#                
#        # use local vars for clarity in the math
#        nifadv, nifcor = self.nifadv, self.nifcor
#        g, f  = self.g, self.f
##        dt, dx = self.dt, self.dx
#        dt    = 0.5*self.dt if self.nt == 1 else self.dt  # must FTCS step for 1st time step (note hb = h set in initialize)
#        dx    = self.dx
#        hb, h = self.hb, self.h
#        ub, u = self.ub, self.u
#        vb, v = self.vb, self.v
#        
#        # calculate FDs
##        dh = self.CS2
#        
#        
#        # h-f(orward) calculated from h-b(ehind) and the current h
#        #   note that (for cyclic BC) h == h[i], hb == hb[i], etc, but h *is* h[i] == False (not same memory loc)
#        # but to do non-cyclic BC, we need the indexing there
#        hf[i] = hb[i] - nifadv * u[i] * dt/dx * (h[ip1]-h[im1]) \
#                      - h[i] * dt/dx * (u[ip1]-u[im1])
#                               
#        # are we allowing winds to change with time?
#        if self.nifwnd:
#            uf[i] = ub[i] - nifadv * u[i] * dt/dx * (u[ip1]-u[im1]) \
#                          - g * dt/dx * (h[ip1]-h[im1]) \
#                          + nifcor * f * v[i] * 2*dt
#        
#            vf[i] = vb[i] - nifadv * u[i] * dt/dx * (v[ip1]-v[im1]) \
#                          - nifcor * f * u[i] * 2*dt                            
#        else:
#            uf, vf = ub, vb
#        
#        # apply explicit numerical diffusion?: 2x FTCS diff2 from n-1
#        if self.nifdif:            
#            dif_factor = self.kdif * 2*dt / dx**2
#                        
#            hf[i] = hf[i] + dif_factor*(hb[ip1]-2*hb[i]+hb[im1])  # 2x FTCS
##            hf[i] = hf[i] + dif_factor*(h[ip1] -2*h[i]+ h[im1])  # CTCS (unstable!)
#            
#            if self.nifwnd:
#                uf[i] = uf[i] + dif_factor*(ub[ip1]-2*ub[i]+ub[im1])
#                vf[i] = vf[i] + dif_factor*(vb[ip1]-2*vb[i]+vb[im1])
#
#        if self.bc == 'open':
#            hf[-1] = hf[-2]
#
#        return hf, uf, vf


    def CTCS2(self):
        """ 2nd-O CTCS (leap-frog) 
        
        h-f(orward) calculated from h-b(ehind) and the current h
          note that (for cyclic BC) h == h[i], hb == hb[i], etc, but h *is* h[i] == False (not same memory loc)
        but to do non-cyclic BC, we need the indexing there
        """
                
#        if self.nt == 1:  # at first time step, have to take a FTCS step
##            return self.FTCS2()
#            dt = self.dt/2
#        else:
#            dt = self.dt
        
        # relevant inds
        i = self.i
        ip1 = self.ip1
        im1 = self.im1
        
        # pre-allocate forward soln
#        hf = np.zeros_like(self.hb)
#        uf = np.zeros_like(self.ub)
#        vf = np.zeros_like(self.vb)
        hf = np.copy(self.h)
        uf = np.copy(self.ub)
        vf = np.copy(self.vb)
                
        # use local vars for clarity in the math
        nifadv, nifcor = self.nifadv, self.nifcor
        g, f  = self.g, self.f
#        dt, dx = self.dt, self.dx
        dt    = 0.5*self.dt if self.nt == 1 else self.dt  # must FTCS step for 1st time step (note hb = h set in initialize)
        dx    = self.dx
        hb, h = self.hb, self.h
        ub, u = self.ub, self.u
        vb, v = self.vb, self.v
        
        # calculate FDs used
        dh = self.CS2d1(h)
        du = self.CS2d1(u)
        dv = self.CS2d1(v)
        d2hb = self.CS2d2(hb)
        d2ub = self.CS2d2(ub)
        d2vb = self.CS2d2(vb)
        
        # for other diffusion schemes:
        d2hb2 = d2hb
        d2hb4 = self.CS4d2(hb)
        d2hb6 = self.CS6d2(hb)
        d4hb2 = self.CS2d4(hb)
        d4hb4 = self.CS4d4(hb)
        d6hb2 = self.CS2d6(hb)
        dif_opts = {'CS2d2': d2hb2,
                    'CS4d2': d2hb4,
                    'CS6d2': d2hb6,
                    'CS2d4': -d4hb2,
                    'CS4d4': -d4hb4,
                    'CS2d6': d6hb2,
                    'none1': 0}
        
#        hten = -nifadv*(u[i]*dh/dx + v[i]*dh/dy) - h[i]*(du/dx + dv/dy)  # dh/dt = F (forcing); could test doing conservative form instead (for 2-D that is)
        
        



        if self.time_filter_scheme == 'Lax-Wendroff':
            # 1-D linear adv only
            
            hstar1 = 0.5*(h[ip1]+h[i]) -0.5*dt*u[i]*(h[ip1]-h[i])/dx  # predicted h at i+1/2, time step nt+1/2
            hstar2 = 0.5*(h[i]+h[im1]) -0.5*dt*u[i]*(h[i]-h[im1])/dx  # predicted h at i-1/2, time step nt+1/2

            hf[i] = h[i] - dt/dx*u[i]*(hstar1-hstar2)  # predicted; CTCS step over 1 dt

        else:
            hten = -nifadv*u[i]*dh/dx - h[i]*du/dx
        
            hf[i] = hb[i] + 2*dt*hten
            

                       
        # are we allowing winds to change with time?
        if self.nifwnd:
#            dh = self.CS6d1(h)
            uten = -nifadv*u[i]*du/dx - g*dh/dx + nifcor*f*v[i]
            
            uf[i] = ub[i] + 2*dt*uten
            
            vten = -nifadv*u[i]*dv/dx - nifcor*f*u[i]                       
    
            vf[i] = vb[i] * 2*dt*vten
            
        else:
            uf, vf = ub, vb
        
        # apply explicit numerical diffusion?: 2x FTCS diff2 from n-1
        if self.nifdif:            
            k = self.kdif
                        
#            hf[i] += 2*dt * k*d2hb/dx**2  # 2x FTCS
            
            scheme_name = self.ctcs_dif_scheme
#            print int(scheme_name[-1])
            delh = dif_opts[scheme_name]
#            print delh
            Dh = delh#/(dx**int(scheme_name[-1]))
#            print Dh
            
            hf[i] += 2*dt * k*Dh
            
            if self.nifwnd:
                uf[i] += 2*dt * k*d2ub/dx**2
                vf[i] += 2*dt * k*d2vb/dx**2

        if self.bc == 'open':
            hf[-1] = hf[-2]

        return hf, uf, vf



    def CTCS4(self):
        """ 2nd-O CiT (leap-frog) for 1-D linear adv equation 
        4th-O CiS
        """
                
        if self.nt == 1:  # at first time step, have to take a FTCS step 
#            return self.FTCScyclic()  # could create a FTCS4 version
            dt = self.dt/2
        else:
            dt = self.dt
        
        # relevant inds
        i = self.i
        ip2 = self.ip2
        ip1 = self.ip1
        im1 = self.im1
        im2 = self.im2
        
        # pre-allocate forward soln
#        hf = np.zeros_like(self.hb)
#        uf = np.zeros_like(self.ub)
#        vf = np.zeros_like(self.vb)
        hf = np.copy(self.hb)
        uf = np.copy(self.ub)
        vf = np.copy(self.vb)
                
        # use local vars for clarity in the math
        nifadv, nifcor = self.nifadv, self.nifcor
        g, f = self.g, self.f
        dx = self.dx
        hb, h = self.hb, self.h
        ub, u = self.ub, self.u
        vb, v = self.vb, self.v
        
        # h-f(orward) calculated from h-b(ehind) and the current h
        hf[i] = hb[i] - nifadv * u[i] * 2*dt/(12*dx) * (-h[ip2]+8*h[ip1]-8*h[im1]+h[im2]) \
                      - h[i] * 2*dt/(12*dx) * (-u[ip2]+8*u[ip1]-8*u[im1]+u[im2])
                               
        # are we allowing winds to change with time?
        if self.nifwnd:
            uf[i] = ub[i] - nifadv * u[i] * 2*dt/(12*dx) * (-u[ip2]+8*u[ip1]-8*u[im1]+u[im2]) \
                          - g * 2*dt/(12*dx) * (-h[ip2]+8*h[ip1]-8*h[im1]+h[im2]) \
                          + nifcor * f * v[i] * 2*dt
        
            vf[i] = vb[i] - nifadv * u[i] * 2*dt/(12*dx) * (-v[ip2]+8*v[ip1]-8*v[im1]+v[im2]) \
                          - nifcor * f * u[i] * 2*dt                            
        else:
            uf, vf = self.ub, self.vb
        
        # apply explicit numerical diffusion?: CTCS4 diff2
        if self.nifdif:            
            dif_factor = self.kdif * 2*dt / (1*dx**2)
                        
            hf[i] = hf[i] + dif_factor*(-hb[ip2]+16*hb[ip1]-30*hb[i]+16*hb[im1]-hb[im2])
            
            if self.nifwnd:
                uf[i] = uf[i] + dif_factor*(-ub[ip2]+16*ub[ip1]-30*ub[i]+16*ub[im1]-ub[im2])
                vf[i] = vf[i] + dif_factor*(-vb[ip2]+16*vb[ip1]-30*vb[i]+16*vb[im1]-vb[im2])

        if self.bc == 'open':
            hf[-2] = hf[-3]
            hf[-1] = hf[-2]
            
            hf[1] = hf[2]
            hf[0] = hf[1]
    
    
        return hf, uf, vf


    def CTCS6(self):
        """ 2nd-O CiT (leap-frog) for 1-D linear adv equation 
        6th-O CiS
        """
                
        if self.nt == 1:  # at first time step, have to take a FTCS step 
            dt = self.dt/2
        else:
            dt = self.dt
        
        # relevant inds
        i = self.i
        ip3 = self.ip3
        ip2 = self.ip2
        ip1 = self.ip1
        im1 = self.im1
        im2 = self.im2
        im3 = self.im3
        
        # pre-allocate forward soln
#        hf = np.zeros_like(self.hb)
#        uf = np.zeros_like(self.ub)
#        vf = np.zeros_like(self.vb)
        hf = np.copy(self.hb)
        uf = np.copy(self.ub)
        vf = np.copy(self.vb)
                
        # use local vars for clarity in the math
        nifadv, nifcor = self.nifadv, self.nifcor
        g, f = self.g, self.f
        dx = self.dx
        hb, h = self.hb, self.h
        ub, u = self.ub, self.u
        vb, v = self.vb, self.v
        
        # h-f(orward) calculated from h-b(ehind) and the current h
        hf[i] = hb[i] - nifadv * u[i] * 2*dt/(60*dx) * (h[ip3]-9*h[ip2]+45*h[ip1]-45*h[im1]+9*h[im2]-h[im3]) \
                      - h[i] * 2*dt/(60*dx) * (u[ip3]-9*u[ip2]+45*u[ip1]-45*u[im1]+9*u[im2]-u[im3])
                               
        # are we allowing winds to change with time?
        if self.nifwnd:
            uf[i] = ub[i] - nifadv * u[i] * 2*dt/(60*dx) * (u[ip3]-9*u[ip2]+45*u[ip1]-45*u[im1]+9*u[im2]-u[im3]) \
                          - g * 2*dt/(60*dx) * (h[ip3]-9*h[ip2]+45*h[ip1]-45*h[im1]+9*h[im2]-h[im3]) \
                          + nifcor * f * v[i] * 2*dt
        
            vf[i] = vb[i] - nifadv * u[i] * 2*dt/(60*dx) * (u[ip3]-9*u[ip2]+45*u[ip1]-45*u[im1]+9*u[im2]-u[im3]) \
                          - nifcor * f * u[i] * 2*dt                            
        else:
            uf, vf = self.ub, self.vb
        
        # apply explicit numerical diffusion?: 2x FTCS6 from n-1; diff2
        if self.nifdif:            
            dif_factor = self.kdif * 2*dt / (1*dx**2)
                        
            d2hb = FDC6d2(hb, im3, im2, im1, i, ip1, ip2, ip3)
            hf[i] = hf[i] + dif_factor*d2hb
            
            if self.nifwnd:
                d2ub = FDC6d2(ub, im3, im2, im1, i, ip1, ip2, ip3)
                d2vb = FDC6d2(vb, im3, im2, im1, i, ip1, ip2, ip3)
                uf[i] = uf[i] + dif_factor*d2ub
                vf[i] = vf[i] + dif_factor*d2vb

        if self.bc == 'open':
            hf[-3] = hf[-4]
            hf[-2] = hf[-3]
            hf[-1] = hf[-2]
            
            hf[2] = hf[3]
            hf[1] = hf[2]
            hf[0] = hf[1]
    
    
        return hf, uf, vf


    def FTCS2(self):
        """ 1st-O FTCS for 1-D linear adv equation """
        
        # relevant inds
        i = self.i
        ip1 = self.ip1
        im1 = self.im1
        
        # pre-allocate forward soln
#        hf = np.zeros_like(self.h)
#        uf = np.zeros_like(self.u)
#        vf = np.zeros_like(self.v)
        hf = np.copy(self.hb)
        uf = np.copy(self.ub)
        vf = np.copy(self.vb)
                
        # use local vars for clarity in the math
        nifadv, nifcor = self.nifadv, self.nifcor
        g, f = self.g, self.f
        dt, dx = self.dt, self.dx
        h = self.h
        u = self.u
        v = self.v
        
        # h-f(orward) calculated from h-b(ehind) and the current h
        hf[i] = h[i] - nifadv * u[i] * dt/(2*dx) * (h[ip1]-h[im1]) \
                     - h[i] * dt/(2*dx) * (u[ip1]-u[im1])
                               
        # are we allowing winds to change with time?
        if self.nifwnd:
            uf[i] = u[i] - nifadv * u[i] * dt/(2*dx) * (u[ip1]-u[im1]) \
                         - g * dt/(2*dx) * (h[ip1]-h[im1]) \
                         + nifcor * f * v[i] * dt
        
            vf[i] = v[i] - nifadv * u[i] * dt/(2*dx) * (v[ip1]-v[im1]) \
                         - nifcor * f * u[i] * dt                            
        else:
            uf, vf = u, v
        
        # apply explicit numerical diffusion?: FTCS diff2
        if self.nifdif:            
            dif_factor = self.kdif * dt / dx**2
                        
            hf[i] = hf[i] + dif_factor*(h[ip1]-2*h[i]+h[im1])
            
            if self.nifwnd:
                uf[i] = uf[i] + dif_factor*(u[ip1]-2*u[i]+u[im1])
                vf[i] = vf[i] + dif_factor*(v[ip1]-2*v[i]+v[im1])
                
        if self.bc == 'open':
            hf[-1] = hf[-2]

        return hf, uf, vf  


    def FTBS(self):
        """ 1st-O FTBS (upstream) for 1-D linear adv equation """
        
        # relevant inds
        i = self.i
        ip1 = self.ip1
        im1 = self.im1
        
        # pre-allocate forward soln
        hf = np.zeros_like(self.h)
        uf = np.zeros_like(self.u)
        vf = np.zeros_like(self.v)
                
        # use local vars for clarity in the math
        nifadv, nifcor = self.nifadv, self.nifcor
        g, f = self.g, self.f
        dt, dx = self.dt, self.dx
        h = self.h
        u = self.u
        v = self.v
        
        # h-f(orward) calculated from h-b(ehind) and the current h
        hf[i] = h[i] - nifadv * u[i] * dt/dx * (h[i]-h[im1]) \
                     - h[i] * dt/dx * (u[i]-u[im1])
                               
        # are we allowing winds to change with time?
        if self.nifwnd:
            uf[i] = u[i] - nifadv * u[i] * dt/dx * (u[i]-u[im1]) \
                         - g * dt/dx * (h[i]-h[im1]) \
                         + nifcor * f * v[i] * dt
        
            vf[i] = v[i] - nifadv * u[i] * dt/dx * (v[i]-v[im1]) \
                         - nifcor * f * u[i] * dt                            
        else:
            uf, vf = u, v
        
        # apply explicit numerical diffusion?: FTCS diff2 for now. should be FTBS?
        if self.nifdif:            
            dif_factor = self.kdif * dt / dx**2
                        
            hf[i] = hf[i] + dif_factor*(h[ip1]-2*h[i]+h[im1])
            
            if self.nifwnd:
                uf[i] = uf[i] + dif_factor*(u[ip1]-2*u[i]+u[im1])
                vf[i] = vf[i] + dif_factor*(v[ip1]-2*v[i]+v[im1])

        return hf, uf, vf


    def RK3CS4(self):
        """ currently just for h 
                
        following Warner pp. 52-53, which comes from Wicker & Skamarock (2002)
        says that the final corrected step should be a CT step
        but the CT version seems considerably more unstable        
        """
        
#        if self.nt == 1:  # at first time step, have to take a FTCS step 
##            return self.FTCScyclic()  # could create a FTCS4 version
#            dt = self.dt/2
#        else:
#            dt = self.dt
        
        dt = self.dt
        
        # relevant inds
        i = self.i
        ip2 = self.ip2
        ip1 = self.ip1
        im1 = self.im1
        im2 = self.im2
        
        # pre-allocate forward soln
        hf = np.zeros_like(self.hb)
        uf = np.zeros_like(self.ub)
        vf = np.zeros_like(self.vb)
                
        # use local vars for clarity in the math
        nifadv, nifcor = self.nifadv, self.nifcor
        g, f = self.g, self.f
        dx = self.dx
        hb, h = self.hb, self.h
        ub, u = self.ub, self.u
        vb, v = self.vb, self.v
        
        # p for prime
        def hten(hp, up): 
            return - nifadv * up[i] * 1/(12*dx) * (-hp[ip2]+8*hp[ip1]-8*hp[im1]+hp[im2]) \
                   - hp[i] * 1/(12*dx) * (-up[ip2]+8*up[ip1]-8*up[im1]+up[im2])

        def uten(hp, up, vp): 
            return - nifadv * up[i] * 1/(12*dx) * (-up[ip2]+8*up[ip1]-8*up[im1]+up[im2]) \
                   - g * 1/(12*dx) * (-hp[ip2]+8*hp[ip1]-8*hp[im1]+hp[im2]) \
                   + nifcor * f * vp[i] * 1
        
        def vten(hp, up, vp):
            return - nifadv * up[i] * 1/(12*dx) * (-vp[ip2]+8*vp[ip1]-8*vp[im1]+vp[im2]) \
                   - nifcor * f * up[i] * 1      

                 
        if self.nifwnd:
            h_star = h + dt/3*hten(h, u)
            u_star = u + dt/3*uten(h, u, v)
            v_star = v + dt/3*vten(h, u, v)
            
            h_dagr = h + dt/2*hten(h_star, u_star)
            u_dagr = u + dt/2*uten(h_star, u_star, v_star)
            v_dagr = v + dt/2*vten(h_star, u_star, v_star)
            
            hf     = h + dt/1*hten(h_dagr, u_dagr)
            uf     = u + dt/1*uten(h_dagr, u_dagr, v_dagr)
            vf     = v + dt/1*vten(h_dagr, u_dagr, v_dagr)

            
        else:
            h_star = h + dt/3*hten(h, u)
            h_dagr = h + dt/2*hten(h_star, u)
            hf     = h + dt/1*hten(h_dagr, u)
            
            uf, vf = u, v
        
        
        return hf, uf, vf


    def RK4CS4(self):
        """ currently just for h 
        
        following https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        """
        
#        if self.nt == 1:  # at first time step, have to take a FTCS step 
##            return self.FTCScyclic()  # could create a FTCS4 version
#            dt = self.dt/2
#        else:
#            dt = self.dt
        
        dt = self.dt
        
        # relevant inds
        i = self.i
        ip2 = self.ip2
        ip1 = self.ip1
        im1 = self.im1
        im2 = self.im2
        
        # pre-allocate forward soln
        hf = np.copy(self.h)
        uf = np.copy(self.u)
        vf = np.copy(self.v)
                
        # use local vars for clarity in the math
        nifadv, nifcor = self.nifadv, self.nifcor
        g, f = self.g, self.f
        dx = self.dx
        hb, h = self.hb, self.h
        ub, u = self.ub, self.u
        vb, v = self.vb, self.v
        
#        # p for prime
#        hten = lambda hp, up: - nifadv * up[i] * 1/(12*dx) * (-hp[ip2]+8*hp[ip1]-8*hp[im1]+hp[im2]) \
#                              - hp[i] * 1/(12*dx) * (-up[ip2]+8*up[ip1]-8*up[im1]+up[im2])
            
        # p for prime
        def hten(hp, up): 
            return - nifadv * up[i] * 1/(12*dx) * (-hp[ip2]+8*hp[ip1]-8*hp[im1]+hp[im2]) \
                   - hp[i] * 1/(12*dx) * (-up[ip2]+8*up[ip1]-8*up[im1]+up[im2])

        def uten(hp, up, vp): 
            return - nifadv * up[i] * 1/(12*dx) * (-up[ip2]+8*up[ip1]-8*up[im1]+up[im2]) \
                   - g * 1/(12*dx) * (-hp[ip2]+8*hp[ip1]-8*hp[im1]+hp[im2]) \
                   + nifcor * f * vp[i] * 1
        
        def vten(hp, up, vp):
            return - nifadv * up[i] * 1/(12*dx) * (-vp[ip2]+8*vp[ip1]-8*vp[im1]+vp[im2]) \
                   - nifcor * f * up[i] * 1   
        
        if self.nifwnd:
            k1h = hten(h, u)
            k1u = uten(h, u, v)
            k1v = vten(h, u, v)
            
            k2h = hten(h + dt/2*k1h, u + dt/2*k1u)
            k2u = uten(h + dt/2*k1h, u + dt/2*k1u, v + dt/2*k1v)
            k2v = vten(h + dt/2*k1h, u + dt/2*k1u, v + dt/2*k1v)
            
            k3h = hten(h + dt/2*k2h, u + dt/2*k2u)
            k3u = uten(h + dt/2*k2h, u + dt/2*k2u, v + dt/2*k2v)
            k3v = vten(h + dt/2*k2h, u + dt/2*k2u, v + dt/2*k2v)
            
            k4h = hten(h + dt/1*k3h, u + dt/1*k3u)
            k4u = uten(h + dt/1*k3h, u + dt/1*k3u, v + dt/1*k3v)
            k4v = vten(h + dt/1*k3h, u + dt/1*k3u, v + dt/1*k3v)
            
            hf = h + dt/6*(k1h + 2*k2h + 2*k3h + k4h) 
            uf = u + dt/6*(k1u + 2*k2u + 2*k3u + k4u) 
            vf = v + dt/6*(k1v + 2*k2v + 2*k3v + k4v) 

        else:
            k1 = hten(h, u)
            k2 = hten(h + dt/2*k1, u)
            k3 = hten(h + dt/2*k2, u)
            k4 = hten(h + dt/1*k3, u)
            hf = h + dt/6*(k1 + 2*k2 + 2*k3 + k4)
            
            uf, vf = u, v
        

        return hf, uf, vf      


    def RK4CS6(self):
        """ currently just for h 
        
        following https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        """
        
        if self.nt == 1:  # at first time step, have to take a FTCS step 
#            return self.FTCScyclic()  # could create a FTCS4 version
            dt = self.dt/2
        else:
            dt = self.dt
        
        # relevant inds
        i = self.i
        ip3 = self.ip3
        ip2 = self.ip2
        ip1 = self.ip1
        im1 = self.im1
        im2 = self.im2
        im3 = self.im3
        
        # pre-allocate forward soln
        hf = np.zeros_like(self.hb)
        uf = np.zeros_like(self.ub)
        vf = np.zeros_like(self.vb)
                
        # use local vars for clarity in the math
        nifadv, nifcor = self.nifadv, self.nifcor
        g, f = self.g, self.f
        dx = self.dx
        hb, h = self.hb, self.h
        ub, u = self.ub, self.u
        vb, v = self.vb, self.v
        
        # p for prime
#        hten = lambda hp, up: - nifadv * up[i] * 1/(60*dx) * (hp[ip3]-9*hp[ip2]+45*hp[ip1]-45*hp[im1]+9*hp[im2]-hp[im3]) \
#                                       - hp[i] * 1/(60*dx) * (up[ip3]-9*up[ip2]+45*up[ip1]-45*up[im1]+9*up[im2]-up[im3])
        
        def hten(hp, up):
            ten = - nifadv * up[i] * 1/(60*dx) * (hp[ip3]-9*hp[ip2]+45*hp[ip1]-45*hp[im1]+9*hp[im2]-hp[im3]) \
                  - hp[i] * 1/(60*dx) * (up[ip3]-9*up[ip2]+45*up[ip1]-45*up[im1]+9*up[im2]-up[im3])
            return ten
        
        # with no nifwnd, uten, vten == 0
        
        # FT
        k1 = hten(h, u)
        k2 = hten(h + dt/2*k1, u)
        k3 = hten(h + dt/2*k2, u)
        k4 = hten(h + dt/1*k3, u)
        hf = h + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        
        # or CT?

        return hf, u, v 


    def spectral(self):
        """
        
        following Durran (2e, 2010) sec. 6.2.1.3: The Equivalent Grid-Point Method
          pp. 289-290
          
        p. 297 could be used for the nonlin. version?
        
        and using 2nd-O CT
        
        for 1-D lin adv, stable for cdt/dx </= 1/pi ~= 0.3183
        
        only for h currently
        """
        
        # relevant inds
        i = self.i
        ip3 = self.ip3
        ip2 = self.ip2
        ip1 = self.ip1
        im1 = self.im1
        im2 = self.im2
        im3 = self.im3
        
        h  = self.h
        u  = self.u
        v  = self.v
        hb = self.hb  # initialize() sets hb = h initially
        dt = self.dt
        c  = self.C
        
        # pre-allocate forward solution
        hf = np.copy(self.hb)
        
#        if self.nt == 1:
#            h_n = np.fft.fft(h, self.Nx)
#        else:
#            h_n = np.fft.fft(hb, self.Nx)
            
        h_n = np.fft.fft(h, self.Nx)
            
#        print h_n.size
        
        N = self.Nx // 2  # there should be 2N+1 grid points
#        L = self.x[-1] - self.x[0]  # domain length (m)
        L = self.Nx * self.dx  # has to be this one to work. if x vals are at center of grid boxes, then this is the total domain size
        
#        n = np.arange(-N, N+1, 1)  # (zonal?) wave numbers
        n = np.hstack((0, np.arange(1, N+1), np.arange(-N, 0)))
#        print n.size
        
        imag = np.complex(0, 1)  # 0 + 1i
        
#        print h_n
#        print n
#        print (n * h_n)[:10] 
#        print (n * h_n * -i)[:10]
        
#        dhdt = np.zeros_like(self.x, dtype=np.complex)
#        for j, x_j in enumerate(self.x):
#            dhdt[j] = np.sum(-i * n*2*np.pi/L * self.C * h_n * np.exp(i*n*2*np.pi/L*x_j))
#            dhdt[j] = np.sum(-i * n*2*np.pi/L * self.C * h_n)
#            dhdt[j] = np.sum(-i * n * self.C * h_n)
#            dhdt[j] = np.sum(-n * self.C * h_n)
        
        
#        dhdt = np.fft.ifft(dhdt)
            
        dhdt = np.fft.ifft(-imag * n*2*np.pi/L * c * h_n)  # works!!
        
#        d2hdt2 = np.fft.ifft(1 * n**2 * (2*np.pi/L)**2 * c * h_n)
        
        
#        print dhdt[:10]
        
        
        if self.nt == 1:
            hf = h + np.real(dhdt) * 1*dt
        
        else:
            hf = hb + np.real(dhdt) * 2*dt
            
            
        if self.nifdif:            
            dif_factor = self.kdif * 2*dt / self.dx**2
            hf[i] = hf[i] + dif_factor*(hb[ip1]-2*hb[i]+hb[im1])
#            hf = hf + 100*np.real(d2hdt2)*dt
            
#            if self.nifwnd:
#                uf[i] = uf[i] + dif_factor*(u[ip1]-2*u[i]+u[im1])
#                vf[i] = vf[i] + dif_factor*(v[ip1]-2*v[i]+v[im1])

        
        return hf, self.u, self.v


    def spectralInterp(self):
        """

        same as spectral() but interpolating between grid points to use more modes
        
        doesn't seem to work better in current form...
        """
        
        h  = self.h
        hb = self.hb  # initialize() sets hb = h initially
        dt = self.dt
        c  = self.C
        x  = self.x
        
        # pre-allocate forward solution
        hf = np.copy(self.hb)
        
        mult = 4
        new_dx = self.dx * 1.0/mult
        
        x_interp = np.arange(x[0], x[-1]+new_dx, new_dx)
#        f = interpolate.interp1d(x, h)
        f = interpolate.CubicSpline(x, h)
        h_interp = f(x_interp)
        
        h_n = np.fft.fft(h_interp)
#        print h_n.size
        
        N = x_interp.size // 2  # there should be 2N+1 grid points
#        L = self.x[-1] - self.x[0]  # domain length (m)
#        L = self.Nx*mult * self.dx  # has to be this one to work. if x vals are at center of grid boxes, then this is the total domain size
        L = x_interp.size * new_dx
        
#        n = np.arange(-N, N+1, 1)  # (zonal?) wave numbers
        n = np.hstack((0, np.arange(1, N+1), np.arange(-N, 0)))
#        print n.size
        
        i = np.complex(0, 1)  # 0 + 1i
            
        dhdt = np.fft.ifft(-i * n*2*np.pi/L * c * h_n)
        
        # get back to the orig grid
#        dhdt_grid = np.full(h.shape, dhdt[0])
#        dhdt_grid[-1] = dhdt[-1]
#        ix_orig = np.array([int(np.where(x_interp == p)[0]) for p in x])[1:-1]
#        dhdt_grid[1:-1] = 1.0/19 * (1*dhdt[ix_orig-2] + 4*dhdt[ix_orig-1] + 9*dhdt[ix_orig] + \
#                         4*dhdt[ix_orig+1] + 1*dhdt[ix_orig+2])
#        f2 = interpolate.interp1d(x_interp, dhdt)
        f2 = interpolate.CubicSpline(x_interp, dhdt)
        dhdt_grid = f2(x)
        
        dhdt = dhdt_grid
        if self.nt == 1:
            hf = h + np.real(dhdt) * 1*dt
        
        else:
            hf = hb + np.real(dhdt) * 2*dt
            

        return hf, self.u, self.v
     

        
if __name__ == '__main__':
    
    m = model(dt=50, dx=1e5, Nt=200, Nx=101,
              hw=13, hc=50, C=10,
              adv_scheme='RK3CS4', bc='cyclic', 
#              time_filter_scheme='RAW', time_filter_nu=0.1, time_filter_alpha=0.5,
              nifadv=1, nifdif=0, nifwnd=1, nifcor=0,
              show_anim=True, anim_stride=10,
              )
    m.run()

    
#    hmax = np.max(m.data['h'], axis=1)
#    plt.plot(hmax)
        
        