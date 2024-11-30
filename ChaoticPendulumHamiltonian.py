# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:59:56 2024

@author: 90545
"""
import numpy as np

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
        
        # initial canonical momentalar eklendi
        self.p1 = (m1 + m2) * (l1**2) * w1 + m2 * l1 * l2 * w2 * np.cos(ang1 - ang2)
        self.p2 = m2 * (l2**2) * w2 + m2 * l1 * l2 * w1 * np.cos(ang1 - ang2)
        
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
        V= -(m1+m2)*l1*cos(theta1)-m2*g*l2*cos(theta2)
        """
        V= -(m1+m2)* g* l1* np.cos(ang1)- m2* g* l2* np.cos(ang2)
        
        return V #Joule
    
    
    def myOmega(self):
        l1 = self.l1
        l2 = self.l2
        m1 = self.m1
        m2 = self.m2
        ang1 = self.ang1
        ang2 = self.ang2
        
        p1 = self.p1
        p2 = self.p2
        
        constant1 = l1 * l2 * (m1 + m2*(np.sin(ang1-ang2))**2 ) 
        
        w1 = (l2*p1 - l1*p2*np.cos(ang1-ang2)) / (l1*constant1)
        w2 = (l1* (m1+m2) * p2 - l2 *m2*p1*np.cos(ang1-ang2)) / (constant1*m2*l2)
        
        return (w1, w2)

    def kinetic_energy(self):
        m1= self.m1
        l1= self.l1
        m2= self.m2
        l2= self.l2
        ang1= self.ang1
        ang2= self.ang2
        
        """
        T= 0.5 * (m1*V1^2 + m2*V2^2)
        X1'= w1*l1*cos(theta1)
        Y1'= w1*l1*sin(theta1)
        X2'= w1*l1*cos(theta1) + w2*l2*cos(theta2)
        Y2'= w1*l1*cosh(theta1) + w2*l2*sin(theta2)
        """
        (w1,w2) = self.omega()
        
        T = 0.5* m1* ( (l1* w1) )**2+ 0.5* m2* ( (l1*w1)**2+ (l2*w2)**2+ 2* l1* l2* w1* w2* np.cos(ang1-ang2) )
        
        return T #Joule
    
    def mechanic_energy(self):
        """ The mechanical energy (total energy). """
        return self.potential_energy() + self.kinetic_energy() #Joule
        
    def hamiltonian(self,ang1,ang2,p1,p2):
        
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2
        g=self.g
        
        constant1 = l1 * l2 * (m1 + m2 * np.sin(ang1 - ang2)**2)
        #h1
        constant2 = (p1 * p2 * np.sin(ang1 - ang2)) / constant1
        #h2
        constant3 = (m2 * (l2 * p1)**2 + (m1 + m2) * (l1 * p2)**2 -
              2 * l1 * l2 * m2 * p1 * p2 * np.cos(ang1 - ang2)) * np.sin(2 * (ang1 - ang2)) / (2 * constant1**2)
        
        w1_= (1/(l1*constant1)) * ( l2*p1-l1*p2*np.cos(ang1-ang2))
        w2_= (1/(l2*m2*constant1)) * (l1*(m1+m2)*p2-l2*m2*p1*np.cos(ang1-ang2))
        p1_=-(m1+m2)*g*l1*np.sin(ang1)-constant2+constant3
        p2_=-m2 * g* l2* np.sin(ang2) + constant2 - constant3
        
        return np.array([w1_, w2_, p1_, p2_])
    
    def Runge_Kutta_4(self,dt):
        
        # Omega ve p'ler.
        f=np.array([self.ang1, self.ang2, self.p1, self.p2])
        
        k1 = self.hamiltonian(*f)
        k2 = self.hamiltonian(*(dt*k1/2 + f))
        k3 = self.hamiltonian(*(dt*k2/2 + f))
        k4 = self.hamiltonian(*(dt*k3 + f))
        
        # w ve g'ler güncellendi. RK4 4 elemanlı array!!!!!
        RK4 = 1.0/6.0 * dt * (k1 + 2.*k2 + 2.*k3 + k4)
        self.ang1 += RK4[0]
        self.ang2 += RK4[1]
        self.p1 += RK4[2]
        self.p2 += RK4[3]
        
pendulum1 = chaotic_pendulum(1, 1, 1, 1, 9.8, 90, 100, 1, 2)
pendulum2 = chaotic_pendulum(1, 1, 1, 1, 9.800001, 90, 100, 1, 2)

import pygame

pygame.init()
screen = pygame.display.set_mode((800,600))
clock = pygame.time.Clock()

pendulum1_bob1_x, pendulum1_bob1_y = [], []
pendulum1_bob2_x, pendulum1_bob2_y = [], []
pendulum2_bob1_x, pendulum2_bob1_y = [], []
pendulum2_bob2_x, pendulum2_bob2_y = [], []



def chaotic_pendulum_simulation(Pendulum1, screen):

    x1,y1 = 400, 225
    x2 = x1 + Pendulum1.l1 * np.sin(Pendulum1.ang1) *100
    y2 = y1 + Pendulum1.l1 * np.cos(Pendulum1.ang1) *100
    x3 = x2 + Pendulum1.l2 * np.sin(Pendulum1.ang2) *100
    y3 = y2 + Pendulum1.l2 * np.cos(Pendulum1.ang2) *100
    
    pygame.draw.line(screen, (0,0,100), (x1, y1), (x2, y2), 2)
    pygame.draw.line(screen, (0,100,0), (x2, y2), (x3, y3), 2)
    pygame.draw.circle(screen, (0,100,100), (int(x2), int(y2)), 12)
    pygame.draw.circle(screen, (100,0,0), (int(x3), int(y3)), 12)  
    
    return x2, y2, x3, y3
   
running = True
WHITE = (255,255,255)
time = []
curr_time = 0
while running:
    screen.fill(WHITE)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    pendulum1.Runge_Kutta_4(dt=1/60)
    pendulum2.Runge_Kutta_4(dt=1/60)
    
    
    chaotic_pendulum_simulation(pendulum1, screen)
    chaotic_pendulum_simulation(pendulum2, screen)
    
    x1_1, y1_1, x1_2, y1_2 = chaotic_pendulum_simulation(pendulum1, screen)
    x2_1, y2_1, x2_2, y2_2 = chaotic_pendulum_simulation(pendulum2, screen)
    
    pendulum1_bob1_x.append(x1_1)
    pendulum1_bob1_y.append(y1_1)
    pendulum1_bob2_x.append(x1_2)
    pendulum1_bob2_y.append(y1_2)

    pendulum2_bob1_x.append(x2_1)
    pendulum2_bob1_y.append(y2_1)
    pendulum2_bob2_x.append(x2_2)
    pendulum2_bob2_y.append(y2_2)
    
    time.append(curr_time)
    curr_time += 1/60
    
    pygame.display.flip()
    clock.tick(60)


pygame.quit()

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 10))

plt.subplot(4, 2, 1)
plt.plot(pendulum1_bob1_x, pendulum1_bob1_y, label="Pendulum 1 - Bob 1")
plt.title("Pendulum 1 - Bob 1 (x-y)")
plt.xlabel("x-coordinate")
plt.ylabel("y-coordinate")
plt.grid(True)
plt.legend()

plt.subplot(4, 2, 2)
plt.plot(pendulum1_bob2_x, pendulum1_bob2_y, label="Pendulum 1 - Bob 2")
plt.title("Pendulum 1 - Bob 2 (x-y)")
plt.xlabel("x-coordinate")
plt.ylabel("y-coordinate")
plt.grid(True)
plt.legend()

plt.subplot(4, 2, 3)
plt.plot(pendulum2_bob1_x, pendulum2_bob1_y, label="Pendulum 2 - Bob 1")
plt.title("Pendulum 2 - Bob 1 (x-y)")
plt.xlabel("x-coordinate")
plt.ylabel("y-coordinate")
plt.grid(True)
plt.legend()

plt.subplot(4, 2, 4)
plt.plot(pendulum2_bob2_x, pendulum2_bob2_y, label="Pendulum 2 - Bob 2")
plt.title("Pendulum 2 - Bob 2 (x-y)")
plt.xlabel("x-coordinate")
plt.ylabel("y-coordinate")
plt.grid(True)
plt.legend()

plt.subplot(4, 2, 5)
plt.plot(time, pendulum1_bob1_x, label="Pendulum 1 - Bob 1 (t-x)")
plt.plot(time, pendulum1_bob1_y, label="Pendulum 1 - Bob 1 (t-y)")
plt.title("Pendulum 1 - Bob 1 (t-x, t-y)")
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.grid(True)
plt.legend()

plt.subplot(4, 2, 6)
plt.plot(time, pendulum2_bob1_x, label="Pendulum 2 - Bob 1 (t-x)")
plt.plot(time, pendulum2_bob1_y, label="Pendulum 2 - Bob 1 (t-y)")
plt.title("Pendulum 2 - Bob 1 (t-x, t-y)")
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.grid(True)
plt.legend()

plt.subplot(4, 2, 7)
plt.plot(time, pendulum1_bob2_x, label="Pendulum 1 - Bob 2 (t-x)")
plt.plot(time, pendulum1_bob2_y, label="Pendulum 1 - Bob 2 (t-y)")
plt.title("Pendulum 2 - Bob 1 (x-t, y-t)")
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.grid(True)
plt.legend()

plt.subplot(4, 2, 8)
plt.plot(time, pendulum2_bob2_x, label="Pendulum 2 - Bob 2 (t-x)")
plt.plot(time, pendulum2_bob2_y, label="Pendulum 2 - Bob 2 (t-x)")
plt.title("Pendulum 2 - Bob 1 (x-t, y-t)")
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.grid(True)
plt.legend()

#billards ball stadium

plt.tight_layout()
plt.show()

ax1 = plt.figure().add_subplot(projection='3d')
ax1.plot(pendulum1_bob1_x, pendulum1_bob1_y, time, label = "1-1")
ax1.plot(pendulum2_bob1_x, pendulum2_bob1_y, time, label = "2-1")

ax1 = plt.figure().add_subplot(projection='3d')
ax1.plot(pendulum1_bob2_x, pendulum1_bob2_y, time, label = "1-2")
ax1.plot(pendulum2_bob2_x, pendulum2_bob2_y, time, label = "2-2")