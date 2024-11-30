# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:16:23 2024

@author: 90545
"""


import numpy as np
from numba import jit

# // 06.11.2024
class chaotic_pendulum:
    
    def __init__(self, l1, l2, m1, m2, g, ang1, ang2, w1, w2):
        """
        l1 : rod-1 length (m)
        l2 : rod-2 length (m)
        m1 : rod-1 mass (kg)
        m2 : rod-2 mass (kg)
        g : the gravitational acceleration (m/s^2)
        ang1, ang2 : initial angles
        w1, w2 : angular velocities (rad/s)
        """
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.g = g
        
        """ Angles must be radians """
        self.ang1 = ang1* (np.pi/ 180)
        self.ang2 = ang2* (np.pi/ 180)
        
        self.w1 = w1 
        self.w2 = w2
        
        
    def potential_energy(self):
        m1= self.m1
        m2= self.m2
        g= self.g
        l1= self.l1
        ang1= self.ang1
        l2= self.l2
        ang2= self.ang2
        
        """
        V= m1*g*Y1 + m2*g*Y2
        Y1= -l1*cos(theta1)
        Y2= -l1*cos(theta1)-l2*cos(theta2)
        V= -(m1+m2)*l1*cos(theta1)-m2g*l2*cos(theta2)
        """
        V= -(m1+m2)* g* l1* np.cos(ang1)- m2* g* l2* np.cos(ang2)
        
        return V #Joule
    
    def kinetic_energy(self):
        m1= self.m1
        l1= self.l1
        w1= self.w1
        m2= self.m2
        l2= self.l2
        w2= self.w2
        ang1= self.ang1
        ang2= self.ang2
        
        
        """
        T= 0.5 * (m1*V1^2 + m2*V2^2)
        X1'= w1*l1*cos(theta1)
        Y1'= w1*l1*sin(theta1)
        X2'= w1*l1*cos(theta1) + w2*l2*cos(theta2)
        Y2'= w1*l1*cosh(theta1) + w2*l2*sin(theta2)
        """
        
        T = 0.5* m1* (l1* w1)**2+ 0.5* m2* ( (l1**2)*w1**2+ (l2**2)*w2**2+ 2* l1* l2* w1* w2* np.cos(ang1-ang2) )
        
        return T #Joule
    
    def mechanic_energy(self):
        """ The mechanical energy (total energy). """
        return self.potential_energy() + self.kinetic_energy() #Joule
            
    def lagrangian(self, ang1,ang2, w1,w2):
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2
        g = self.g
        
        """
        Lagrangian kullanılarak 2 tane nonlinear diff denklem bulundu. 
        Linear Algebra kullanılarak theta1'' ve theta2'' bulundu.
        Sonrasında ise w1, w2, theta1'' ve theta2'' array döndürüldü.
        """
        
        """
        theta1''+ (alpha1)*theta2'' = f1
        theta2'' + (alpha2)*theta1'' = f2
        """
        """
        A.(theta1'') = (f1)  ==> (theta1'') = A**(-1).(f1)
          (theta2'')   (f2)      (theta2'')           (f2)
          
        A = (A11, A12   =>> A**(-1) = 1/det(A) * ( A11,-A21
             A21, A22)                            -A12, A22 )
        """
        
        alpha1 = (l2/l1) * (m2/(m1+m2))*np.cos(ang1-ang2)
        alpha2 = (l1/l2) * np.cos(ang1-ang2)
        
        f1 = -(l2/l1) * (m2/(m1+m2)) * np.sin(ang1-ang2)*(w2**2) - g*np.sin(ang1)/l1
        f2 = (l1/l2) * np.sin(ang1-ang2)*(w1**2) - g*np.sin(ang2)/l2
        
        g1 = (f1 - alpha1*f2) / (1 - alpha1*alpha2) # theta1''
        g2 = (-alpha2*f1 + f2) / (1 - alpha1*alpha2) # theta2''
        
        return np.array([w1, w2, g1, g2])
        
    def Runge_Kutta_4(self,dt):
        
        # Açılar ve Omega'lar
        f=np.array([self.ang1, self.ang2, self.w1, self.w2])
        
        
        k1 = self.lagrangian(*f)
        k2 = self.lagrangian(*(f+dt*k1/2))
        k3 = self.lagrangian(*(f+dt*k2/2))
        k4 = self.lagrangian(*(f+dt*k3))
        
        # w ve g'ler güncellendi. RK4 4 elemanlı array
        RK4 = 1.0/6.0 * dt * (k1 + 2.0*k2 + 2.0*k3 + k4)
        self.ang1 += RK4[0]
        self.ang2 += RK4[1]
        self.w1 += RK4[2]
        self.w2 += RK4[3]
        
        # Hata kontrolü
        if not np.isfinite(self.ang1) or not np.isfinite(self.ang2):
            raise ValueError("Numerical instability detected in Runge-Kutta step.")
        


pendulum1 = chaotic_pendulum(1, 1, 1, 1, 9.8, 90, 100, 1, 2)
pendulum2 = chaotic_pendulum(1, 1, 1, 1, 9.80001, 90, 100, 1, 2)

import pygame

pygame.init()
screen = pygame.display.set_mode((800,600))
clock = pygame.time.Clock()


def chaotic_pendulum_simulation(Pendulum1, Pendulum2, screen):
    x1,y1 = 400, 225
    x2 = x1 + Pendulum1.l1 * np.sin(Pendulum1.ang1) *100
    y2 = y1 + Pendulum1.l1 * np.cos(Pendulum1.ang1) *100
    x3 = x2 + Pendulum1.l2 * np.sin(Pendulum1.ang2) *100
    y3 = y2 + Pendulum1.l2 * np.cos(Pendulum1.ang2) *100
    
    pygame.draw.line(screen, (0,0,100), (x1, y1), (x2, y2), 2)
    pygame.draw.line(screen, (0,100,0), (x2, y2), (x3, y3), 2)
    pygame.draw.circle(screen, (0,100,100), (int(x2), int(y2)), 12)
    pygame.draw.circle(screen, (100,0,0), (int(x3), int(y3)), 12)  
    
    z1,t1 = 400, 225
    z2 = z1 + Pendulum2.l1 * np.sin(Pendulum2.ang1) *100
    t2 = t1 + Pendulum2.l1 * np.cos(Pendulum2.ang1) *100
    z3 = z2 + Pendulum2.l2 * np.sin(Pendulum2.ang2) *100
    t3 = t2 + Pendulum2.l2 * np.cos(Pendulum2.ang2) *100
    
    pygame.draw.line(screen, (100,0,100), (z1, t1), (z2, t2), 2)
    pygame.draw.line(screen, (100,100,0), (z2, t2), (z3, t3), 2)
    pygame.draw.circle(screen, (100,100,100), (int(z2), int(t2)), 12)
    pygame.draw.circle(screen, (100,0,0), (int(z3), int(t3)), 12)  

running = True
WHITE = (255,255,255)
while running:
    screen.fill(WHITE)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    pendulum1.Runge_Kutta_4(dt=1/60)
    pendulum2.Runge_Kutta_4(dt=1/60)
    
    chaotic_pendulum_simulation(pendulum1, pendulum2, screen)
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()