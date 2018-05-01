# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 17:54:40 2016

@author: pbarros
RECALCULAR ESFORCOS
ARRUMAR COORDENADAS EM RELACAO AOS EIXOS DA ROM E DO NAVIO
"""

import numpy as np
np.set_printoptions(precision = 4)
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
import math 
import rotation
import newtonRaphson2 as nR2
import gaussPivot as gauss


#Objeto navio
class navio:
    def __init__(self, nav_data, div=5, dx=0.0, dy=1.6757135024103853, dth=0.0, pc1=1/3, pc2=9/24, ph=0.2):
        """Cria objeto navio, com os parametros passados. Os valores pc1, pc2 e
        ph indicam o inicio da curvatura e dos chanfros"""        
        self.div = div
        self.t = np.arange(self.div)/self.div
        self.L = nav_data.loc['Loa']
        self.H = nav_data.loc['B']
        self.T = nav_data.loc['T']
        self.dx = dx        
        self.dy = dy
        self.dth = dth
        self.c1 = self.L*pc1
        self.c2 = self.L*pc2
        self.h2 = self.H*ph
        self.Rz = rotation.matrix([0,0,1],dth)               

        self.LWT =  0.50*nav_data.loc['Delta_m'] 
        self.Md = nav_data.loc['Delta_m'] #self.LWT + self.DWT
        self.D = nav_data.loc['De']
        self.G = self.T - self.D 
        self.z = nav_data.loc['KG'] - self.D
        
    def pontos(self):
        """Desenha a geometria do navio"""
        
        self.sc = 1.        
        self.x = self.sc*np.array([-155., -139.4, -124., -108.5, -93., -77.5, -62., -46.5, -31., -15.5, 0,    15.5, 31.,  46.5, 62.,  77.5, 93.,  108.5, 124.,  139.5, 155.])
        self.y = self.sc*np.array([ 9.23,  14.37, 18.98,  23.6,  23.6, 23.6,  23.6, 23.6,  23.6, 23.6, 23.6, 23.6, 23.6, 23.6, 23.6, 23.6, 23.6, 23.6,  21.55, 14.37, 3.59])
        self.px_index = len(self.x)
        #self.py_index = len(self.x)/2

        self.coord = np.array([self.x,self.y,np.full(len(self.x),self.z)])
        
        self.x = self.x[::-1]
        self.y = -self.y[::-1]        
        self.new = np.array([self.x,self.y,np.full(len(self.x),self.z)])
        self.coord = np.array([np.append(self.coord[0],self.new[0]),np.append(self.coord[1],self.new[1]),np.append(self.coord[2],self.new[2])])
        self.coord = np.array([np.append(self.coord[0],self.coord[0,0]),np.append(self.coord[1],self.coord[1,0]),np.append(self.coord[2],self.coord[2,0])])

        self.coord[0] = self.coord[0] - (np.amax(self.coord[0])+np.amin(self.coord[0]))/2
        self.coord[1] = self.coord[1] + (np.amax(self.coord[1])-np.amin(self.coord[1]))/2      
       
        self.coordi = np.array(self.coord)
                     
        self.cg = np.array([0 + self.dx, self.H/2 + self.dy, self.z]) 
        self.cgi = np.array(self.cg)
                
        self.thi = 0. + self.dth        
        self.th = float(self.thi)      
       
        self.coordnav(self.dx,self.dy,self.dth)
        
        

    def coordnav(self,dx,dy,dth):
        """Atualiza as coordenadas obtidas na funcao pontos aplicando deslocamentos
        dx, dy e dth (rotacao)"""        
        self.cg_ant = np.array(self.cg)
        self.coord[0] = self.coord[0] - self.cg[0]
        self.coord[1] = self.coord[1] - self.cg[1]        
        self.Rz = rotation.matrix([0,0,1],dth)
        self.coord = np.dot(self.Rz,self.coord)
       
        self.coord[0] = self.coord[0] + self.cg[0] + dx
        self.coord[1] = self.coord[1] + self.cg[1] + dy       
    
        self.px = self.coord[:,self.px_index]
        self.Bx = self.px-self.cg       
        self.basex = self.Bx/math.sqrt(np.dot(self.Bx,self.Bx))
    
    def amarras(self, xai, yai, zai):
        """Define os pontos de amarracao no navio"""               

        self.xai = xai
        self.yai = yai       
        self.zai = zai
        
        self.a = np.array([self.xai,self.yai,self.zai])
        self.at_a(0.0,0.0,self.dth)
    
    def at_a (self, dx, dy, dth):
        """atualiza os pontos de amarracao no navio em funcao do cg e de uma matriz de rotacao"""        
        self.xa = self.a[0] - self.cg_ant[0]
        self.ya = self.a[1] - self.cg_ant[1]
        self.za = self.a[2]
        self.Rz = rotation.matrix([0,0,1],dth)        
        
        self.aux = np.array([self.xa,self.ya,self.za])         
        self.aux = np.dot(self.Rz,self.aux)
        
        self.aux[0] = self.aux[0] + self.cg_ant[0] + dx
        self.aux[1] = self.aux[1] + self.cg_ant[1] + dy       

        self.a = np.array([self.aux[0],self.aux[1],self.aux[2]])               
        
    
    def contato(self, def_data):
        """Define os pontos de contato da defensa com o navio"""        
        self.xdi = def_data[0]
        self.ydi = np.zeros(len(self.xdi))
        self.zdi = def_data[2]  

        self.yd = np.zeros(len(self.xdi))
        self.at_b()
        self.ydi = self.yd
    
    def at_b (self):
        """atualiza os pontos de contato no navio em funcao do cg e de uma matriz de rotacao"""
        self.argc = int((len(n.coord[0]))/2)
        self.pts_con = np.array(self.coord[:,self.argc:len(n.coord[0])])

        self.xd = self.xdi
        self.zd = self.zdi      
            
        for i, x in enumerate(self.xdi):
            self.aux_con = self.pts_con[0] - x            
            self.arg1 = np.argmin(abs(self.aux_con))             
            
            if (self.aux_con[self.arg1] < 0 and self.arg1 == 0) or (self.aux_con[self.arg1] > 0 and self.arg1 == len(self.aux_con)-1):
                self.yd[i] = 99999.
                #print(self.yd[i],self.arg1)
                #print(self.aux_con)
            
            elif (self.aux_con[self.arg1] > 0 and self.aux_con[self.arg1+1] > self.aux_con[self.arg1]): #(self.aux_con[self.arg1] < 0 and self.aux_con[self.arg1-1] > self.aux_con[self.arg1]) or 
                self.yd[i] = 99999.
                #print(self.yd[i],self.arg1)
                #print(self.aux_con)
            
            elif self.aux_con[self.arg1] < 0:
                #print(self.arg1)
                self.arg1 = self.arg1 + self.argc
                self.arg2 = self.arg1 - 1
                self.yd[i] = self.coord[1,n.arg1] + (x-self.coord[0,n.arg1])*(self.coord[1,n.arg2]-self.coord[1,n.arg1])/(self.coord[0,n.arg2]-self.coord[0,n.arg1])
                #print(self.yd[i],self.arg1,self.arg2)
                #print(self.aux_con)

            elif self.aux_con[self.arg1] > 0:
                #print(self.arg1)                
                self.arg1 = self.arg1 + self.argc
                self.arg2 = self.arg1 + 1
                self.yd[i] = self.coord[1,n.arg1] + (x-self.coord[0,n.arg1])*(self.coord[1,n.arg2]-self.coord[1,n.arg1])/(self.coord[0,n.arg2]-self.coord[0,n.arg1])            
                #print(self.yd[i],self.arg1,self.arg2)
                #print(self.aux_con)
                
            #print('Defensa {0}\n{1}: {2}\n{3}: {4}'.format(i,self.arg1,self.aux_con[self.arg1],self.arg2,self.aux_con[self.arg2]))        
        
        #self.yd = self.yd
        self.b = np.array([self.xd,self.yd,self.zd])
        #self.b.loc[:,('y')] = self.b.loc[:,('y')] 
    
        
#Objeto linhas de amarracao
class linha:
    #Funcao Constructor
    def __init__(self, i, p, n, diam = 92e-3, mat ='UHMWPE SK75', MBL=578.34*9.8066*1e3, eu=4.5E-2, FS=2.0):
        #propriedades da secao e do material        
        self.i = i        
        self.mat = mat
        self.diam = diam        
        self.A = math.pi*self.diam**2/4  
        
        self.eu = eu
        self.FS = FS
        self.MBL = MBL
        self.Fu = self.MBL/self.FS
        self.su = self.Fu/self.A
        self.E = self.MBL/(self.A*self.eu)
        
        #self.s0 = 0.1*self.su/self.FS
        self.F0 = 0.05*self.MBL
        
        #geometria inicial
        self.p = np.array(p[:,self.i])
        self.pi = self.p[0:3]
        self.pf = self.p[3:6]               
        
        self.geom(self.pf[0],self.pf[1],self.pf[2])
        self.Li = self.L/(1+self.F0/(self.E*self.A))     
        self.vi = self.v
        #forca inicial
        self.forca()
        self.K = self.E*self.A/self.L #somente para calculo do modelo 1 da ROM
    
    #Funcao atualiza geometria    
    def geom(self, xf, yf, zf):
        self.Vx = xf - self.pi[0]    
        self.Vy = yf - self.pi[1]
        self.Vz = zf - self.pi[2]
        self.V = np.array([self.Vx,self.Vy,self.Vz])
          
        self.L = math.sqrt(self.Vx**2 + self.Vy**2 + self.Vz**2)
        
        self.v = self.V/self.L
        self.vx = self.v[0]
        self.vy = self.v[1]
        self.vz = self.v[2]
        
        self.th = math.atan(self.vy/self.vx)
        self.th_rel = self.th - math.acos(n.basex[0])       
        self.tv = math.asin(self.vz)
        
        self.v2 = n.cg - self.pi
        self.proj = np.vdot(self.v2,self.v)*self.v
        self.brac = (self.proj + self.pi) - n.cg    
    
    #Funcao calcula forcas
    def forca(self):
        self.DL = self.L - self.Li        
        if self.DL > 0:
            self.F_escalar = self.DL/self.Li*self.E*self.A
        else:
            self.F_escalar = 0.0
        self.F = -(self.F_escalar)*self.v
        self.M = np.cross(self.brac,self.F)

        self.M_Fx = np.cross(self.brac,np.array([self.F[0],0.,0.]))
        self.M_Fy = np.cross(self.brac,np.array([0.,self.F[1],0.]))

    def verificacao(self):
        if self.F_escalar > self.Fu:
            print('\nERRO: Tracao na linha {0} excede máximo valor permitido.\nMBL = {1:.2f} kN\nF_  = {2:.2f} kN\nFu  = {3:.2f} kN\nFS = {4}'.format(self.i,self.MBL/1E3,self.F_escalar/1E3,self.Fu/1E3,self.FS))
            return False
        if self.DL < 0:
            print('\nERRO: Linha {0} nao esta tracionada.\nDL = {1:.2f} m'.format(self.i,self.DL))
            return False
        else:
            return True
            
    def data(self):         
        #print('xi={0}\nyi={1}\nxf={2}\nyf={3}'.format(self.xi,self.yi,self.xf,self.yf))
        #print(self.p)
        #print(self.p.loc['tipo'])        
        print('Comprimento = {0:.2f}m\nTeta hor ={1:.2f} graus\nTeta ver ={2:.2f} graus'.format(
        self.L,math.degrees(self.th),math.degrees(self.tv)))
        
        #print('Teta hor. rel. {0:.2f} graus'.format(math.degrees(self.th_rel)))
        #print('Direção \n',self.v)        

        #print('F_ult {0:.2f} kN'.format(self.F_ult/1000))
        #print('Rigidez {0:.2f} kN/m'.format(self.K/10**3))
        #print('Força\n{0}'.format(self.F/1E3))
        #print('Braco \n{0}'.format(self.brac))
        #print('Momento\n{0}'.format(self.M/1E3))
        #print('Fx = {0:.3f} kN\nFy = {1:.3f} kN \nMz = {2:.3f} kNm\n'.format(self.F.loc['x']/1E3,self.F.loc['y']/1E3,self.M.loc['z']/1E3))        
        
        
#Objeto defensas
class defensa:
    #Funcao Constructor
    def __init__(self, i, p, n, tipo='SCN13000 E3.1', R = 2107e3, E = 1463e3, eu = -0.7, ep=-0.2, h=1.3, mi=0, pan=0.4):
        self.i = i        
        self.p = p[:,self.i]
        self.h = h
        self.eu = eu 
        self.ep = ep
        self.x = self.p[0]   
        self.y = self.p[1]
        self.z = self.p[2]       
        self.pan = pan
        self.R = R
        self.E = E
        self.K = 0.75*self.R/(-self.ep*self.h)
        self.mi = mi
        

        self.forca()        
    
    def forca(self):
        self.yf = n.b[1,self.i]
        self.Dh = (self.yf-self.pan) - self.h
        self.e = self.Dh/self.h
        
        if self.e > 0:
            self.Fy = 0.0
        else:
            self.Fy = -self.Dh*self.K 
        
        self.Fx = self.mi*self.Fy
        self.Fz = self.mi*self.Fy
        
        self.F = np.array([self.Fx,self.Fy,self.Fz]) 

        if self.Fy != 0:
            self.v = self.F/math.sqrt(np.sum(self.F*self.F))
            self.v2 = n.cg - self.p
            self.proj = np.vdot(self.v2,self.v)*self.v
            self.brac = (self.proj + self.p) - n.cg   
            self.M = np.cross(self.brac,self.F)
        else:
            self.M = np.zeros(len(self.F))

    def verificacao(self):
        if self.e < self.eu:
            print('\nERRO: Deformacao na defensa {0} excede máximo valor permitido.\ne_ = {1:.2f} kN\neu = {2:.2f} kN'.format(self.i,self.e,self.eu,))
            return False         
        else:
            return True
        
    def data(self):          
        #print('\nBraco [m]\n{0}'.format(self.brac))
        #print('Forca [kN]\n{0}'.format(self.F/1E3))
        #print('\nMomento [kNm]\n{0}'.format(self.M/1E3))
        print('Fx = {0:.3f} kN\nFy = {1:.3f} kN \nMz = {2:.3f} kNm\n'.format(self.F.loc['x']/1E3,self.F.loc['y']/1E3,self.M.loc['z']/1E3))

#Funcao calcula a forca resultante do sistema (Fext + Fint) para uma dada posicao do CG.
#A posicao do CG muda apos cada iteracao
def f(x):        
    pi = np.array([n.cg[0],n.cg[1],n.th])         
    #print(pi)
    n.cg[0] = x[0]
    n.cg[1] = x[1]
    n.th = x[2]  
    
    #atualiza o desenho do navio a partir das coordenadas iniciais
    pf = np.array(x)
    dist = np.array(pf-pi)
    n.coordnav(dist[0],dist[1],dist[2])     
    n.at_a(dist[0],dist[1],dist[2])
    n.at_b()
    SF = np.zeros(len(x))
    for i, li in enumerate(l):
        li.pf[0] = n.a[0,i]
        li.pf[1] = n.a[1,i]      
        li.geom(li.pf[0],li.pf[1],li.pf[2])        
        li.forca()        
        SF = SF + np.array([li.F[0],li.F[1],li.M[2]])
    for di in d:
        di.forca()        
        SF = SF + np.array([di.F[0],di.F[1],di.M[2]])  
    return SF + f_ext
    

#Funcao calcula a forca resultante do sistema (Fext + Fint) para uma dada posicao do CG.
#A posicao do CG é restaurada apos cada iteracao
def f2(x):        
    pi = np.array([n.cg[0],n.cg[1],n.th])         
    coordi = np.array(n.coord)    
    ai = np.array(n.a)
    bi = np.array(n.b)    
    #print(pi)
    n.cg[0] = x[0]
    n.cg[1] = x[1]
    n.th = x[2]  
    
    #atualiza o desenho do navio a partir das coordenadas iniciais
    pf = np.array(x)
    dist = np.array(pf-pi)
    n.coordnav(dist[0],dist[1],dist[2])     
    n.at_a(dist[0],dist[1],dist[2])
    n.at_b()
    SF = np.zeros(len(x))
    for i, li in enumerate(l):
        li.pf[0] = n.a[0,i]
        li.pf[1] = n.a[1,i]      
        li.geom(li.pf[0],li.pf[1],li.pf[2])        
        li.forca()        
        SF = SF + np.array([li.F[0],li.F[1],li.M[2]])
    for di in d:
        di.forca()        
        SF = SF + np.array([di.F[0],di.F[1],di.M[2]])  
    
    n.cg[0] = float(pi[0])
    n.cg[1] = float(pi[1])
    n.th = float(pi[2])
    n.coord = np.array(coordi)
    n.a = np.array(ai)
    n.b = np.array(bi)
    
    return SF + f_ext


#funcao produz curvas de reacao em funcao da posicao a partir do ponto de equilibrio como origem
def RvsP():
    
    #CUIDADO: A ENORME QUANTIDADE DE TRANSFORMACOES QUE ESTA FUNCAO FAZ GERA ERROS NUMERICOS
        
    Rx = pd.DataFrame({'x':(),'Fx':(),'Fy':(),'Mz':()})
    for i in np.arange(-.1, .1, 0.01):    
        delta = np.array([i,0.,0.])
        rx = f2(pos_f+delta)
        bo = True
        for li in l:
            bo = bool(bo * li.verificacao())
        for di in d:
            bo = bool(bo*di.verificacao())        
        if bo == True:
            Rx = Rx.append(pd.Series({'x':i,'Fx':rx[0],'Fy':rx[1],'Mz':rx[2]}),ignore_index=True)
    plt_f(Rx['x'],-Rx['Fx']/1000.,xlabel='Deslocamento em x (m)', ylabel ='Forca (kN)',titulo='Forca Fx')
    plt_f(Rx['x'],-Rx['Fy']/1000.,xlabel='Deslocamento em x (m)', ylabel ='Forca (kN)',titulo='Forca Fy')
    plt_f(Rx['x'],-Rx['Mz']/1000.,xlabel='Deslocamento em x (m)', ylabel ='Momento (kNm)',titulo='Momento Mz')
    

    Ry = pd.DataFrame({'y':(),'Fx':(),'Fy':(),'Mz':()})
    for j in np.arange(-0.2, 1., 0.01):    
        delta = np.array([0.,j,0.])
        ry = f2(pos_f+delta)
        bo = True
        """        
        for li in l:
            bo = bool(bo * li.verificacao())
        for di in d:
            bo = bool(bo*di.verificacao())        
        if bo == True:
        """
        Ry = Ry.append(pd.Series({'y':j,'Fx':ry[0],'Fy':ry[1],'Mz':ry[2]}),ignore_index=True)
    plt_f(Ry['y'],-Ry['Fx']/1000.,xlabel='Deslocamento em y (m)', ylabel ='Forca (kN)',titulo='Forca Fx')
    plt_f(Ry['y'],-Ry['Fy']/1000.,xlabel='Deslocamento em y (m)', ylabel ='Forca (kN)',titulo='Forca Fy')
    plt_f(Ry['y'],-Ry['Mz']/1000.,xlabel='Deslocamento em y (m)', ylabel ='Momento (kNm)',titulo='Momento Mz')
    
    
    
    Rth = pd.DataFrame({'th':(),'Fx':(),'Fy':(),'Mz':()})
    for k in np.arange(-0.000001, 0.000001, 0.0000001): 
        delta = np.array([0.,0.,k])        
        rth = f2(pos_f+delta)
        bo = True
        for li in l:
            bo = bool(bo * li.verificacao())
        for di in d:
            bo = bool(bo*di.verificacao())        
        if bo == True:
            Rth = Rth.append(pd.Series({'th':k,'Fx':rth[0],'Fy':rth[1],'Mz':rth[2]}),ignore_index=True)
        
    plt_f(Rth['th']*1e3,-Rth['Fx']/1000.,xlabel='Rotacao (10^-3 rad)', ylabel ='Forca (kN)',titulo='Forca Fx')
    plt_f(Rth['th']*1e3,-Rth['Fy']/1000.,xlabel='Rotacao (10^-3 rad)', ylabel ='Forca (kN)',titulo='Forca Fy')
    plt_f(Rth['th']*1e3,-Rth['Mz']/1000.,xlabel='Rotacao (10^-3 rad)', ylabel ='Momento (kNm)',titulo='Momento Mz')
    
    return Rx,Ry,Rth
   
#Funcao plotar geometria
def plt_geo(n,l,d):       
    fig, ax = plt.subplots(facecolor='white', figsize=(20, 6), dpi = 60)       
    fig.suptitle('Geometria', fontsize=20, fontweight='bold', y =0.8)
    fig.subplots_adjust(hspace=0.5,  top = 0.8) 
    ax.grid(True)

    for li in l:
        x = [li.pi[0], li.pf[0]]
        y = [li.pi[1], li.pf[1]]
        """xf e yf sao apenas para a geometria inicial para a configuracao deformada, usar n.xa!!!!"""        
        
        plt.plot(x , y, 'b-', linewidth=1.0, marker = 'o',markersize=15)

    for di in d:
        x = di.x
        y = di.y
        plt.plot(x , y, 'r-', linewidth=1.0, marker = '^',markersize=15)
    
    x = n.coord[0]
    y = n.coord[1]
    plt.plot(x , y, 'k-', linewidth=1.0)
    
    x_max = 200
    x_min = -200
    y_max = 70
    y_min = -10
    
    x = (x_min, x_max)   
    y = (0, 0)
    plt.plot(x , y, 'k-', linewidth=1.0)
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect('equal')


#Funcao plotar grafico
def plt_f(x,y,xlabel='',ylabel='',titulo =''):
    fig, ax = plt.subplots(facecolor='white') 
    fig.suptitle(titulo, fontsize=20, fontweight='bold')    
    ax.grid(True) 
    plt.plot(x , y, 'k-', linewidth=1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


#Esforco de Vento
def vento(Vvt, alfa, AVT, AVL, CDVT, CDVL, KVe, L, roa=1.23):
    
    fi = math.atan((AVL/AVT)*math.tan(alfa))
    RV = (roa/2*Vvt**2)*(CDVL*AVT*(math.cos(alfa))**2+CDVT*AVL*(math.sin(alfa))**2)/math.cos(fi-alfa)
    FVL = RV*math.cos(fi)
    FVT = RV*math.sin(fi)
    MCGV = FVT*KVe*L   
    
    FV = pd.Series(np.array([FVL,FVT,MCGV]),index=('FVL','FVT','MCGV'))
    return fi, RV, FV


#Esforco de Corrente - Pressao
def corr_p(Vct, alfa, ACT, ACL, CDCL, CDCT, KCe, L, row=1025.):
    
    fi = math.atan((ACL/ACT)*math.tan(alfa))
    RC = (row/2*Vct**2)*(CDCL*ACT*(math.cos(alfa))**2+CDCT*ACL*(math.sin(alfa))**2)/math.cos(fi-alfa)
    FCL = RC*math.cos(fi)
    FCT = RC*math.sin(fi)
    MCGC = FCT*KCe*L

    FC = pd.Series(np.array([FCL,FCT,MCGC]),index=('FCL','FCT','MCGC'))
    return fi, RC, FC


#Esforco de Corrente - Arrasto
def corr_f(Vct, alfa, AfCT, AfCL, CfC, L, row=1025.):  
    #fi = math.atan((AfCT/AfCL)*(math.tan(alfa))**2)
    FfCL = row/2*(Vct**2)*CfC*AfCL*(math.cos(alfa))**2
    FfCT = row/2*(Vct**2)*CfC*AfCT*(math.sin(alfa))**2
    MCGfC = 0.0

    FC = pd.Series(np.array([FfCL,FfCT,MCGfC]),index=('FfCL','FfCT','MCGfC'))
    return FC


#Cria todos os casos de vento e corrente
def casos():
    vvt = [31.]
    calv = [[nav_data.loc['ATe'],nav_data.loc['ALe']]]
    caso= ['De']
    angv = np.array([0, 10, 30, 60, 90, 120, 150, 180])
    cdvl = 0.9-angv/180.*0.2
    cdvt = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    kve = [0., -0.1/3., -0.10, -0.12, -0.16, -0.27, -0.37, 0.]
    tabv = pd.DataFrame(index=())


    for v in vvt:
        for j, c in enumerate(calv):    
            for i, a in enumerate(angv):                    
                phi, RV, Fvento = vento(Vvt=v, alfa=math.pi*a/180, CDVL=cdvl[i], CDVT=cdvt[i], KVe=kve[i], AVT=c[0], AVL=c[1], L=nav_data['Loa'])
                line = pd.Series((caso[j],v,a,180*phi/math.pi,cdvl[i],cdvt[i],kve[i], RV, Fvento.loc['FVL'], Fvento.loc['FVT'], Fvento.loc['MCGV']),
                                 index=('caso','Vvt','Ang','Phi','CDVL','CDVT','Kve','Rv', 'FL', 'FT','MCG'))
                tabv = tabv.append(line, ignore_index=True)

    vc = [1.54, 1.03, 0.39]
    calc = [[nav_data.loc['ATs'],nav_data.loc['ALs']]]
    calf = [[nav_data.loc['ATfs'],nav_data['ALfs']]]
    angc = [0.,10., 90.]
    cdcl = [0.6]
    cdct = [5.5]
    #cdct = 3.0 para h/D =1.9 ---- profundidade da agua considerada: 15m00m
    #v = 0.97*10^-6 m^2/s 
    #cfc = 0.075/(log Re - 2)^2 ____ usando o minimo de 0.001, ou trocando Lpp por B para corrente transversal
    #cfc = 0.001 para buque nuevo
    #cfc = 0.004 para buque em servicio  
    #Re = Vc * Lpp * cos(angc)/v
    cfc = [0.004]

    kce = [0., 0.06, 0.]
    tabp = pd.DataFrame(index=())
    tabf = pd.DataFrame(index=())

    for j, c in enumerate(calc):
        for i, a in enumerate(angc):
            phi, RC, Fcorr_p = corr_p(Vct=vc[i], alfa=math.pi*a/180, CDCT=cdct[j], CDCL=cdcl[j], KCe=kce[i], ACT=calc[j][0], ACL=calc[j][1], L=nav_data['Loa'])
            line = pd.Series((caso[j],a,180*phi/math.pi, RC,Fcorr_p.loc['FCL'], Fcorr_p.loc['FCT'], Fcorr_p.loc['MCGC'],vc[i]),
                             index=('caso','Ang_c','Phi','RC','FL', 'FT','MCG','Vc'))
            tabp = tabp.append(line, ignore_index=True)
        
            Fcorr_f = corr_f(Vct=vc[i], alfa=math.pi*a/180, CfC=cfc[0], AfCT=calf[j][0], AfCL=calf[j][1],  L=nav_data['Loa'])        
            line = pd.Series((caso[j],a,Fcorr_f.loc['FfCL'], Fcorr_f.loc['FfCT'], Fcorr_f.loc['MCGfC'],vc[i]),
                             index=('caso','Ang_c','FL', 'FT','MCG','Vc'))
            tabf = tabf.append(line, ignore_index=True)

    return tabv, tabp, tabf

#Calcula esforcos pelo metodo 2
def metodo2():
    tabv, tabp, tabf = casos()
    #print('\nVento\n',tabv)
    #print('\nPressao\n',tabp)
    #print('\nAtrito\n',tabf)

    #Analise da hipotese 3 para os springs

    hip = pd.DataFrame()
    for i in tabp.index:
        indices = tabv[tabv['caso']==tabp.loc[i,'caso']].index
        hip = hip.append(tabv.loc[indices,('FL','FT','MCG','caso')])
        hip.loc[indices,('FL','FT','MCG')] = hip.loc[indices,('FL','FT','MCG')] + tabp.loc[i,'FL':'MCG'] + tabf.loc[i,'FL':'MCG']
        hip.loc[indices,('Ang_v')] = tabv.loc[indices,('Ang')]
        hip.loc[indices,('Vvt')] = tabv.loc[indices,('Vvt')]
        hip.loc[indices,('Ang_c')] = tabp.loc[i,('Ang_c')]
        hip.loc[indices,('Vc')] = tabp.loc[i,('Vc')]
        hip.index = ['ant']*len(hip)#np.arange(tabp.index.max()*len(hip),(tabp.index.max()+1)*len(hip))


    lista = [2,3]#lin_data[lin_data['tipo']=='Spring'].index.values
    hip['FL_ver'] = 0.
    for i in lista:               
        flin = (-hip.loc[:,('FL')]/l[i].vi[0]*gd)
        flin[flin < 0] = 0.
        flin.name = 'B'+str(i)    
        hip[flin.name]=flin
        hip.loc[:,('FL_ver')] = hip.loc[:,('FL_ver')] + hip.loc[:,flin.name]*l[i].vi[0]/gd

    hip.index = np.arange(len(hip))  


    lista = [0,1,4,5]#lin_data[(lin_data['tipo']=='Traves') | (lin_data['tipo']=='Lancante')].index.values
    K = 0.
    J = 0.
    for i in lista:        
        K = K + l[i].vi[1]
        J = J + l[i].vi[1]*abs(l[i].pf[0])
    hip['FT_ver'] = 0.
    hip['MCG_ver'] = 0.
    fext = hip.loc[:,('FT')]
    for i in lista:
        ft = fext/K*gd   
        if l[i].pf[0] < 0:
            fm = -hip.loc[:,('MCG')]/J*gd
        else:
            fm = hip.loc[:,('MCG')]/J*gd
        flin = (ft + fm)    
        hip['FT_ver'] = hip['FT_ver'] + ft*l[i].vi[1]/gd
        hip['MCG_ver'] = hip['MCG_ver'] - fm*l[i].vi[1]*l[i].pf[0]/gd
        flin.name = 'B'+str(i)  
        hip[flin.name]=flin
    
    """
    np.set_printoptions(precision=5,suppress=True)
    #print(hip.loc[:,('Vvt','Ang_c','Ang_v','FL','FT','MCG')]/1e3)
    hip = hip.sort_index(axis=1)
    print(hip) #[(hip['Ang_v']==0) | (hip['Ang_v']==10) | (hip['Ang_v']==90)]
  
  
    print('\n')
    print(tabv)
    print('\n')
    print(tabp)
    print('\n')
    print(tabf)
    """

    #writer = pd.ExcelWriter('metodo.xlsx')
    #hip.to_excel(writer,'Sheet1')
 

    """
    writer = pd.ExcelWriter('hipotese.xlsx')
    hip.to_excel(writer,'Sheet2')

    #print('\n',hip.loc[:,('Ang_v','FL','FL_ver','FT','FT_ver','MCG','MCG_ver')])
    #print(hip)

    writer = pd.ExcelWriter('resumo.xlsx')
    resumo.loc[:,('Fx','Fy','Mz','Fh')] = resumo.

    w = hip.loc[:,('B0','B1','B2','B3','B4','B5')]/1e3
    w['Ang_v'] = hip.loc[:,'Ang_v']
    resumo.to_excel(writer,'Sheet1')

    Sumitomo 800H X150
    R = 502 kN (70% deflection)
    E = 319 kNm
    K = E/x^2


    for i,li in enumerate(l):      
        print('\nDados de linha {0}:'.format(i))     
        li.data()

    Os comprimentos das linhas de amarracao devem estar entre 35 e 50 metros
    Traveses th = 90 +- 15 graus, (angulo relativo com o eixo longitudinal do navio)
            tv = 0 +-25 graus
            
    Springs th = 0 +- 10 graus
            tv = 0 +-25 graus

    Lancantes th = 45 +- 15 graus sempre se afastando do navio (para fora no cais)
            tv = 0 +-25 graus

    """

def interpolacao(Ry):
    grau = 10
    list = [0]

    for i in range(1,grau+1,1):
        list.append(i)
    
    #coef = np.zeros(grau)
    A = np.full(len(Ry),1.)

    for i in list[1:]:
        A = np.vstack((A,Ry.loc[:,'y'].transpose()**(i)))

    b = gauss.gaussPivot(A.dot(A.transpose()),A.dot(-Ry.loc[:,'Fy']))
    print(b)    
    
    for j,i in enumerate(list):
        C = b[j]*Ry.loc[:,'y']**i
        Ry['C'+str(i)] = C

    Ry['est'] = Ry.loc[:,'C0':'C'+str(grau)].sum(axis = 1)
    print(Ry.loc[:,('Fy','est')])


    fig, ax = plt.subplots(facecolor='white') 

    xlabel='Deslocamento em y (m)'
    ylabel ='Forca (kN)'
    titulo='Forca Fy'

    fig.suptitle(titulo, fontsize=20, fontweight='bold')    
    ax.grid(True) 

    plt.plot(Ry['y'] , -Ry['Fy']/1000., 'k-', linewidth=1.0)
    plt.plot(Ry['y'] , Ry['est']/1000., 'r-', linewidth=1.0)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
#_________________________________________________________________________________________________________
     

#Início do código
nav_data = pd.Series({'Type':'VLCC',
                      'Delta_m':235000, #DWT = aprox. 0.875*Delta_m = 273.5
                      #DWT adotado 275000
                      #LWT = Md - DWT = 39.000
                      #Carregamento em lastro: Delta_lastro = 0.25*DWT = 68.375
                      #Cb*(D_lastro*B*Lpp_lastro) = LWT + Delta_lastro = 107.375
                      #Lpp p/ 8 metros acima da quilha = 300m
                      #D_lastro = 107375/(0.8089*300*58) = 7.63 
                      #D_lastro adotado: 8.0m
                      
                      'Loa':330.,
                      'Lpp':310.,
                      'B':47.2, 
                      'T':28.5, #Pontal
                      'D':18.9, #Calado de projeto (máximo)
                      
                      #Parametros para navio em lastro
                      #Cb = 0.8098
                      'De':18.9, #no caso minimo: aprox 0.4*D (OK)
                      
                      #Ver tabela 4.6.4.33 da ROM - p.240 para aproximar valores de 
                      # hT = 24.00 na ROM 3.1______ 1875/58-9.2 = 23.13m
                      # hL = 3.70 na ROM 3.1_______7655/320-9.2 = 14.72m
                      #Francobordo: G = T - D = 22.00
                      #Comprimento na linha d'agua: Lif
                      #Largura na linha d'agua Bif
                      #Calado no porto 15
                      #h/D = 1,875 --- CDCT = 3.0
                      
                      'ATe':47.2*(28.5-18.9+21), #B*(G+hT)=1444.32
                      'ALe':310*(28.5-18.9+3.4), #Lpp*(G+hL)=4030. 
                      'ATs':47.2*18.9,  #Bif*D=892.08
                      'ALs':310*18.9, #Lif*D=5859.
                      'ATfs':(310+2*18.9)*47.2, #(Lpp+2D)*B=16416.
                      'ALfs':(47.2+2*18.9)*310, #(B+2D)*Lpp=26350.                 
                      
                      'KG':13.32
                      }) 
 

#define a geometria do navio
n = navio(nav_data)
n.pontos()

n.amarras(np.array([-155, -155, -80, 80, 155, 155]),
          np.array([25.3, 25.3, 1.7, 1.7, 25.3, 25.3]),       
          np.full(6,n.T-n.D))

#define as linhas de amarracao
lin_data = np.array([[-180., -156., -48., 48., 156., 180], np.full(6, -2.0), np.full(6,4.05),
                     n.a[0], n.a[1], n.a[2]])
l = [linha(i, lin_data, n) for i in range(len(lin_data[0,:]))] #cria objetos linha
#print(n.coord)

#define as defensas
s_def = 24 #espaçamento
nd = 3#int(n.L/(2*s_def)) #quantidade a partir do cg do navio
def_data = np.array([np.array(range(-nd,nd+1))*s_def,np.zeros(2*nd+1),np.full(2*nd+1,2.0)])
n.contato(def_data) #pontos de contato com o navio
d = [defensa(i, def_data, n) for i in range(len(def_data[0,:]))] #cria objetos defensa


#define vetores de forcas externas
ext_data = pd.DataFrame({'Nome':('Vento','Corrente'), 'Fx':(0.,0.), 'Fy':(0.,0.), 'Mz':(0.,0.)})
gd = 1.5 #amplificacao dinamica
f_ext = 1e3*gd*np.array(np.sum(ext_data.loc[:,('Fx','Fy','Mz')]))                                 
f_ext[0] = -f_ext[0] #correcao do esforco na figura da ROM para os eixos globais

#encontra posicao de equilibrio
P = np.array([n.cg[0],n.cg[1],n.th])

pos_f = nR2.newtonRaphson2(f,P,h=1E-12,tol=1E-15,max_it=30)

#plota geometria final
#plt_geo(n,l,d) 

#organiza e plota saida final de forcas por objeto
resumo = pd.DataFrame({'Elemento':(),'Fx':(),'Fy':(),'Mz':()})
for i, li in enumerate(l):
    resumo = resumo.append(pd.Series({'Elemento':'Linha '+str(i),'Fx':li.F[0],'Fy':li.F[1],'Fz':li.F[2],'Mz':li.M[2],'F':li.F_escalar, 'Mz_Fx':li.M_Fx[2], 'Mz_Fy':li.M_Fy[2]}),ignore_index=True)
for i, di in enumerate(d):
    resumo = resumo.append(pd.Series({'Elemento':'Defensa '+str(i),'Fx':di.F[0],'Fy':di.F[1],'Fz':di.F[2],'Mz':di.M[2]}),ignore_index=True)
resumo['Fh'] = np.sqrt(resumo.loc[:,'Fx']**2+resumo.loc[:,'Fy']**2)
resumo.loc[(resumo.loc[:,'Mz_Fx'] != 0) & (~np.isnan(resumo.loc[:,'Mz_Fx'])),'Mz_Fx/Mz'] = resumo.loc[(resumo.loc[:,'Mz_Fx'] != 0) & (~np.isnan(resumo.loc[:,'Mz_Fx'])),'Mz_Fx']/resumo.loc[(resumo.loc[:,'Mz_Fx'] != 0) & (~np.isnan(resumo.loc[:,'Mz_Fx'])),'Mz']

print(resumo)

"""
print('\n\nSumMz(Fx) = {0:.2f} kN'.format(np.sum(resumo.loc[:,'Mz_Fx'])/1e3))
print('\n\nSumMz(Fy) = {0:.2f} kN'.format(np.sum(resumo.loc[:,'Mz_Fy'])/1e3))
print('\n\nSumMz_lin = {0:.2f} kN'.format(np.sum(resumo.loc[0:5,'Mz'])/1e3))
print('\n\nMz(Fx)/Mz =',np.sum(resumo.loc[:,'Mz_Fx'])/np.sum(resumo.loc[:,'Mz']))
"""

#confere somatoria de esforcos internos (deve ser igual a soma dos externos)
print('\n\nSoma dos esforços internos [kN,m]:\n{0}'.format(np.sum(resumo.loc[:,('Fx','Fy','Mz')])/1E3))

#define curvas de forca em funcao da posicao

for li in l:
    li.verificacao()
for di in d:
    di.verificacao()




#writer = pd.ExcelWriter('resumo.xlsx')
#resumo.loc[:,('Fx','Fy','Fz','F','Mz','Fh')] = resumo.loc[:,('Fx','Fy','Fz','F','Mz','Fh')]/1e3
#resumo.to_excel(writer,'Sheet1')

#metodo2()

#n.cgi[0] = float(pos_f[0])
#n.cgi[1] = float(pos_f[1])
#n.thi = float(pos_f[2])
#n.ai = np.array(n.a)
#n.bi = np.array(n.b)
#n.coordi = pd.DataFrame(n.coord)

#Rx, Ry, Rth = RvsP()
#Ry['y']=Ry['y']/(25.3-pos_f[1]) 
#interpolacao(Ry)

#x = np.array(Ry.loc[:,'y'])
#y = np.array(Ry.loc[:,'Fy'])

"""
Rigidez transversal das linhas de amarracao:
a = 0.
for i in l:
    a = i.vy**2*i.K/1e3 + a
    print(i.K, i.vy, i.vy**2*i.K/1e3)
    
b = 0.
for j in d:
    b = j.K/1e3 + b
    print(j.K/1e3)
"""