#!/usr/bin/python3
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import * 
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import math as ma
import sys
import os
import ctypes

from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem

class GLTextItem(GLGraphicsItem):
    def __init__(self, X=None, Y=None, Z=None, text=None):
        GLGraphicsItem.__init__(self)

        self.text = text
        self.X = X
        self.Y = Y
        self.Z = Z

    def setGLViewWidget(self, GLViewWidget):
        self.GLViewWidget = GLViewWidget

    def setText(self, text):
        self.text = text
        self.update()

    def setX(self, X):
        self.X = X
        self.update()

    def setY(self, Y):
        self.Y = Y
        self.update()

    def setZ(self, Z):
        self.Z = Z
        self.update()

    def paint(self):
        self.GLViewWidget.qglColor(QtCore.Qt.white)
        self.GLViewWidget.renderText(self.X, self.Y, self.Z, self.text)

class MAIN(QtGui.QWidget):

    def __init__(self):
        super(MAIN, self).__init__() 
        self.initUI()

    def initUI(self):
    ## Start Data
        global pts
        three = (3,3)
        pts = np.empty(three, dtype = np.int32)

        def cart2sp(x, y, z):
            """Converts data from cartesian coordinates into spherical.

            Args:
                x (scalar or array_like): X-component of data.
                y (scalar or array_like): Y-component of data.
                z (scalar or array_like): Z-component of data.

            Returns:
                Tuple (r, theta, phi) of data in spherical coordinates.
            x = np.asarray(x)
            y = np.asarray(y)
            z = np.asarray(z)
            scalar_input = False
            """
            if x.ndim == 0 and y.ndim == 0 and z.ndim == 0:
                x = x[None]
                y = y[None]
                z = z[None]
                scalar_input = True
            r = np.sqrt(x**2+y**2+z**2)
            theta = np.arccos(z/r)
            phi = np.arctan2(y, x)
            if scalar_input:
                return (r.squeeze(), theta.squeeze(), phi.squeeze())
            return (r, theta, phi)

        def sp2cart(r, theta, phi):
            """Converts data in spherical coordinates into cartesian.

            Args:
                r (scalar or array_like): R-component of data.
                theta (scalar or array_like): Theta-component of data.
                phi (scalar or array_like): Phi-component of data.

            Returns:
                Tuple (x, y, z) of data in cartesian coordinates.
            """
            r = np.asarray(r)
            theta = np.asarray(theta)
            phi = np.asarray(phi)
            scalar_input = False
            if r.ndim == 0 and theta.ndim == 0 and phi.ndim == 0:
                r = r[None]
                theta = theta[None]
                phi = phi[None]
                scalar_input = True
            x = r*np.cos(theta)*np.cos(phi)
            y = r*np.cos(theta)*np.sin(phi)
            z = r*np.sin(theta)
            if scalar_input:
                return (x.squeeze(), y.squeeze(), z.squeeze())
            return (x, y, z)

        def cart2cyl(x, y, z):
            """Converts data in cartesian coordinates into cylyndrical.

            Args:
                x (scalar or array_like): X-component of data.
                y (scalar or array_like): Y-component of data.
                z (scalar or array_like): Z-component of data.

            Returns:
                Tuple (r, phi, z) of data in cylindrical coordinates.
            """
            x = np.asarray(x)
            y = np.asarray(y)
            z = np.asarray(z)
            scalar_input = False
            if x.ndim == 0 and y.ndim == 0 and z.ndim == 0:
                x = x[None]
                y = y[None]
                z = z[None]
                scalar_input = True
            r = np.sqrt(x**2+y**2)
            phi = np.arctan2(y, x)
            if scalar_input:
                return (r.squeeze(), phi.squeeze(), z.squeeze())
            return (r, phi, z)

        def cyl2cart(r, phi, z):
            """Converts data in cylindrical coordinates into cartesian.

            Args:
                r (scalar or array_like): R-component of data.
                phi (scalar or array_like): Phi-component of data.
                z (scalar or array_like): Z-component of data.

            Returns:
                Tuple (x, y, z) of data in cartesian coordinates.
            """
            r = np.asarray(r)
            phi = np.asarray(phi)
            z = np.asarray(z)
            scalar_input = False
            if r.ndim == 0 and phi.ndim == 0 and z.ndim == 0:
                r = r[None]
                phi = phi[None]
                z = z[None]
                scalar_input = True
            x = r*np.cos(phi)
            y = r*np.sin(phi)
            if scalar_input:
                return (x.squeeze(), y.squeeze(), z.squeeze())
            return (x, y, z)        
        
        def cart_conversion():
            global pts
            ## MAIN LOGIC ##
            if pts.size:
                # Check Cartesian
                if not pts[0,:].size == 0:  #If cart array is empty -> Return True
                    print("\nCartesian:", pts[0,0], pts[0,1], pts[0,2])

                if pts[1,:].size == 0:
                    print("Cylindrical Coordinates are Empty")
                else:
                    #print("Cylindrical:", cart2cyl(pts[0,0], pts[0,1], pts[0,2]))
                    R,Phi,Zc = cart2cyl(pts[0,0], pts[0,1], pts[0,2])

                    pts[1,0] = R
                    pts[1,1] = Phi
                    pts[1,2] = Zc

                    R_int.setValue(R)
                    Phi_int.setValue(Phi)
                    Zc_int.setValue(Zc)
                    self.update()

                if pts[2,:].size == 0:
                    print("Spherical Coordinates are Empty")
        
                else:
                    Rho, Theta, PhiS = cart2sp(pts[0,0], pts[0,1], pts[0,2])

                    Rho_int.setValue(Rho)
                    Theta_int.setValue(Theta)
                    PhiS_int.setValue(PhiS)

                    self.update()

        def cyl_conversion():
            global pts
            if pts.size:
                if not pts[1,:].size == 0:
                    print("\nCylindrical:", pts[1,0], pts[1,1], pts[1,2])
                    # Check Spherical

                if pts[0,:].size == 0:
                    print("\nCartesian Coordinates are Empty")
                else:
                    x,y,z = cyl2cart(pts[1,0], pts[1,1], pts[1,2])
                    
                    X_int.setValue(x)
                    Y_int.setValue(y)
                    Z_int.setValue(z)


                    self.update()
              

                if pts[2,:].size ==0:  #If sph array is empty -> Return True
                    print("Spherical Coordinates are Empty")
                else:
                    X,Y,Z = cyl2cart(pts[1,0], pts[1,1], pts[1,2])
                    Rho, Theta, Phi = cart2sp(X,Y,Z)

                    Rho_int.setValue(Rho)
                    Theta_int.setValue(Theta)
                    PhiS_int.setValue(Phi)

                self.update()

            else:   
                print('Unsuccessful')
        
        def sph_conversion():
            global pts
            if pts.size:
                if not pts[2,:].size == 0:
                    print("\n Spherical:", pts[2,0] ,pts[2,1],pts[2,2])
                    # Check Cylindrical
                
                if pts[0,:].size == 0:
                    print("\nCartesian Coordinates are Empty")
                else:
                    print(pts[2,0],pts[2,1],pts[2,2])
                    x,y,z = sp2cart(pts[2,0],pts[2,1],pts[2,2])
                    print(x,y,z)                        
                    X_int.setValue(x)
                    Y_int.setValue(y)
                    Z_int.setValue(z)

                self.update()
 
                if pts[2,:].size == 0:  #If cyl array is empty -> Return True
                    print("Cylindrical Coordinates are Empty")
                else:
                    x,y,z = sp2cart(pts[2,0], pts[2,1], pts[2,2])
                    r,phi,z = cart2cyl(x,y,z)

                    R_int.setValue(r)
                    Phi_int.setValue(phi)
                    Zc_int.setValue(z)

                    self.update()


            else:   
                print('Unsuccessful')

        self.resize(1600,1000)
        self.setWindowTitle('Cartesian - Cylindrical - Spherical - Conversion Calculator')
      
        #Plot
        plot = gl.GLViewWidget()
        plot.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        plot.opts['distance'] = 40

        #Cartesian
        X_int = QtGui.QDoubleSpinBox()
        X_int.setDecimals(4)
        X_int.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
        X_int.setMaximum(1000000)
        #X_int.setValue(pts[0,0])
        X_int.setMinimum(-1000000)

        def Xcallback_int(X_int):
            global pts
            pts[0,0] = (X_int)
            return pts[0,0]

        X_int.valueChanged[float].connect(Xcallback_int)
        X_int.show()
        
        Y_int = QtGui.QDoubleSpinBox()
        Y_int.setDecimals(4)

        Y_int.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
        Y_int.setMaximum(1000000)
        #Y_int.setValue(pts[0,1])
        Y_int.setMinimum(-1000000)

        def Ycallback_int(Y_int):
            global pts
            pts[0,1] = (Y_int)
            return pts[0,1]
                    
        Y_int.valueChanged[float].connect(Ycallback_int)
        Y_int.show()


        Z_int = QtGui.QDoubleSpinBox()
        Z_int.setDecimals(4)
        Z_int.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
        Z_int.setMaximum(1000000)
        Z_int.setMinimum(-1000000)
        #Z_int.setValue(pts[0,2])

        def Zcallback_int(Z_int):
            global pts
            pts[0,2] = (Z_int)
            return pts[0,2]
            
        Z_int.valueChanged[float].connect(Zcallback_int)
        Z_int.show()

        #Cartesian Button
        cart_button = QPushButton("Convert Cartesian", self)
        cart_button.clicked.connect(cart_conversion)
        

        #Cylindrical
        R_int = QtGui.QDoubleSpinBox()
        R_int.setDecimals(4)
        R_int.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
        R_int.setMaximum(1000000)
        R_int.setMinimum(-1000000)
        #R_int.setValue(pts[1,0])

        def R_callback_int(R_int):
            global pts
            pts[1,0] = (R_int)
            return pts[1,0]

        R_int.valueChanged[float].connect(R_callback_int)
        R_int.show()


        Phi_int = QtGui.QDoubleSpinBox()
        Phi_int.setDecimals(4)
        Phi_int.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
        Phi_int.setMaximum(1000000)
        #Phi_int.setValue(pts[1,1])
        Phi_int.setMinimum(-1000000)

        def Phi_callback_int(Phi_int):
            global pts
            pts[1,1] = (Phi_int)
            return pts[1,1]

        Phi_int.valueChanged[float].connect(Phi_callback_int)

        Phi_int.show()

        Zc_int = QtGui.QDoubleSpinBox()
        Zc_int.setDecimals(4)
        Zc_int.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
        Zc_int.setMaximum(1000000)
        #Zc_int.setValue(pts[1,2])
        Zc_int.setMinimum(-1000000)


        def Zc_callback_int(Zc_int):
            global pts
            pts[1,2] = (Zc_int)
            return pts[1,2]

        Zc_int.valueChanged[float].connect(Zc_callback_int)

        Zc_int.show()

        #Cylindrical Button
        cyl_button = QPushButton("Convert Cylindrical", self)

        #Spherical
        Rho_int = QtGui.QDoubleSpinBox()
        Rho_int.setDecimals(4)
        Rho_int.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
        Rho_int.setMaximum(1000000)
        Rho_int.setMinimum(-1000000)
        #Rho_int.setValue(pts[2,0])

        def Rho_callback_int(Rho_int):
            global pts
            pts[2,0] = (Rho_int)
            return pts[2,0]

        Rho_int.valueChanged[float].connect(Rho_callback_int)

        Rho_int.show()

        Theta_int = QtGui.QDoubleSpinBox()
        Theta_int.setDecimals(4)
        Theta_int.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
        Theta_int.setMaximum(1000000)
        #Theta_int.setValue(pts[2,1])
        Theta_int.setMinimum(-1000000)

        def Theta_callback_int(Theta_int):
            global pts
            pts[2,1] = (Theta_int)
            return pts[2,1]

        Theta_int.valueChanged[float].connect(Theta_callback_int)
        Theta_int.show()

        PhiS_int = QtGui.QDoubleSpinBox()
        PhiS_int.setDecimals(4)
        PhiS_int.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
        PhiS_int.setMaximum(1000000)
        #PhiS_int.setValue(pts[2,2])
        PhiS_int.setMinimum(-1000000)

        def PhiS_callback_int(PhiS_int):
            global pts
            pts[2,2] = (PhiS_int)
            return pts[2,2] 

        PhiS_int.valueChanged[float].connect(PhiS_callback_int)
        PhiS_int.show()

        #Spherical Button
        sph_button = QPushButton("Convert Spherical", self)

        def Grids():# Grids
            
            gx = gl.GLGridItem()
            gx.rotate(90, 0, 1, 0)
            gx.translate(0, 0, 0)
            gx.setSpacing(x=1,y=1,z=1)
            plot.addItem(gx)
            
            gy = gl.GLGridItem()
            gy.rotate(90, 1, 0, 0)
            gy.setSpacing(x=1,y=1,z=1)
            plot.addItem(gy)
            
            gz = gl.GLGridItem()
            gz.rotate(0, 0, 0, 0)
            gz.setSpacing(x=1,y=1,z=1)
            plot.addItem(gz)
            

            # Grid Labels
            x = GLTextItem(X=11, Y=0, Z=0, text="X")
            neg_x = GLTextItem(X=-11, Y=0, Z=0, text="-X")
            
            y = GLTextItem(X=0, Y=10, Z=0, text="Y")
            neg_y = GLTextItem(X=0, Y=-11, Z=0, text="-Y")

            z = GLTextItem(X=0, Y=0, Z=11, text="Z")
            neg_z = GLTextItem(X=0, Y=0, Z=-11, text="-Z")

            #calc = GLTextItem(X=1, Y=1, Z=1, text="Calculated Point")
            
            # Positive X
            x.setGLViewWidget(plot)
            plot.addItem(x)
            
            # Negative X
            neg_x.setGLViewWidget(plot)
            plot.addItem(neg_x)
            
            # Positive Y
            y.setGLViewWidget(plot)
            plot.addItem(y)

            # Negative Y
            neg_y.setGLViewWidget(plot)
            plot.addItem(neg_y)

            # Positive Z
            z.setGLViewWidget(plot)
            plot.addItem(z)
                        
            # Negative Z
            neg_z.setGLViewWidget(plot)
            plot.addItem(neg_z)
            plot.setBackgroundColor(96,96,96)

            self.update()
            # Calculated Point
            #calc.setGLViewWidget(plot)
            
            #plot.addItem(calc)
        Grids()
        
        def plot_point():
            # Coordinate Pts.
            post = np.empty((1, 3))
            post[0] = (pts[0,0],pts[0,1],pts[0,2])

 
            co1 = gl.GLScatterPlotItem(pos=post,color = (255,0,0,1), size = 0.5, pxMode = False)
            plot.addItem(co1)



            
        def xyz():
            # Coordinate Pts.
            x0_vector = (0,0,0)
            x_vector = (2,0,0)
            
            y0_vector = (0,0,0)
            y_vector = (0,2,0)
            
            z0_vector = (0,0,0)
            z_vector = (0,0,2)

            xdot = np.array([x0_vector,x_vector])
            ydot = np.array([y0_vector,y_vector])
            zdot = np.array([z0_vector,z_vector])

            co = gl.GLLinePlotItem(pos=xdot, width = 3, color = (255,0,0,1))
            plot.addItem(co)

            co1 = gl.GLLinePlotItem(pos=ydot, width = 3, color = (0,255,0,1))
            plot.addItem(co1)

            co2 = gl.GLLinePlotItem(pos=zdot, width = 3, color = (0,0,255,1))
            plot.addItem(co2)

        xyz()
        
        def super_clear():
            three = (3,3)
            pts = np.zeros(three, dtype = np.int32)
            
            X_int.setValue(0)
            Y_int.setValue(0)
            Z_int.setValue(0)
            R_int.setValue(0)
            Phi_int.setValue(0)
            Zc_int.setValue(0)
            Rho_int.setValue(0)
            Theta_int.setValue(0)
            PhiS_int.setValue(0)

            self.update()


        ## Create a grid layout to manage the widgets size and position
        layout = QtGui.QGridLayout()
        self.setLayout(layout)

        ## Cartesian
        XLabel = QLabel('X:')
        layout.addWidget(XLabel, 1, 0)
        XLabel.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)

        YLabel = QLabel('Y:')
        layout.addWidget(YLabel, 2, 0)
        YLabel.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)

        ZLabel = QLabel('Z:')
        layout.addWidget(ZLabel, 3, 0)
        ZLabel.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)

        layout.addWidget(cart_button, 4, 1, alignment = QtCore.Qt.AlignRight)


        #Cylindirical
        R_Label = QLabel("\u03C1:")
        layout.addWidget(R_Label, 5, 0)
        R_Label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)

        Phi_Label = QLabel('\u03C6:')
        layout.addWidget(Phi_Label, 6, 0)
        Phi_Label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)

        Zc_Label = QLabel('Z:')
        layout.addWidget(Zc_Label, 7, 0)
        Zc_Label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
        
        cyl_button.setCheckable(False)
        cyl_button.toggle()
        
        cyl_button.clicked.connect(cyl_conversion)
        layout.addWidget(cyl_button, 8, 1, alignment = QtCore.Qt.AlignRight)

        #Spherical 
        Rho_Label = QLabel('R:')
        layout.addWidget(Rho_Label, 9, 0)
        Rho_Label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)

        Theta_Label = QLabel('\u03B8:')
        layout.addWidget(Theta_Label, 10, 0)
        Theta_Label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)

        PhiS_Label = QLabel('\u03C6:')
        layout.addWidget(PhiS_Label, 11, 0)
        PhiS_Label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
        
        sph_button.setCheckable(False)
        sph_button.toggle()

        sph_button.clicked.connect(sph_conversion)
        layout.addWidget(sph_button, 12, 1, 1, 1)

        plot_button = QPushButton("Plot Point", self)
        plot_button.setCheckable(False)
        plot_button.toggle()
        plot_button.clicked.connect(plot_point)

        layout.addWidget(plot_button, 15, 1, 1, 1)

        def VCyl():
            global pts
            
            n = 50
            phi = np.linspace(0,pts[1,1],n)
            for i in range(n):
                x = np.array(pts[1,0]/2*np.cos(phi))
                y = np.array(pts[1,0]/2*np.sin(phi))
                z = np.zeros((0,50))
                pts_c = np.vstack([x,y,z]).transpose()
                plt = gl.GLLinePlotItem(pos=pts_c, color = (1, 1, 1, 1), width=2)
                plot.addItem(plt)

            x,y,z = cyl2cart(pts[1,0], pts[1,1], pts[1,2])

            theta_line_data = np.array(([0,0,0],[x,y,z])) 
            theta_line = gl.GLLinePlotItem(pos=theta_line_data, width=(2), antialias=True)
            plot.addItem(theta_line)
            
            new1 = np.array(([0,0,0],[x,y,0]))
            plt1 = gl.GLLinePlotItem(pos=new1, width=(2), antialias=True)
            plot.addItem(plt1)
            
            
            new2 = np.array(([x/2,y/2,0],[x/2,y/2,z/2])) 
            plt2 = gl.GLLinePlotItem(pos=new2, width=(2), antialias=True)
            plot.addItem(plt2)
            

            md = gl.MeshData.cylinder(rows=20, cols=20, radius=[pts[1,0], pts[1,0]], length=pts[1,2], offset = False)
            m5 = gl.GLMeshItem(meshdata=md, color = (0, 0, 0, 0.5), smooth=True, glOptions='translucent')
            plot.addItem(m5)


            phi = pts[1,1]
            # Grid Labels
            Rho_Grid_Label = GLTextItem(X=(pts[1,0]*np.cos(phi)+0.05), Y=(pts[1,0]*np.sin(phi)+0.05), Z=0, text="\u03C1")
            
            Phi_Grid_Label = GLTextItem(X=(pts[1,0]/2*np.cos(phi/2)+0.05), Y=(pts[1,0]/2*np.sin(phi/2)+0.05), Z=0, text="\u03C6")

            Zc_Grid_Label = GLTextItem(X=x/2+0.05, Y=y/2+0.05, Z=z/4, text="Z")

          
            # Positive X
            Rho_Grid_Label.setGLViewWidget(plot)
            plot.addItem(Rho_Grid_Label)
            
            # Positive Y
            Phi_Grid_Label.setGLViewWidget(plot)
            plot.addItem(Phi_Grid_Label)

            # Positive Z
            Zc_Grid_Label.setGLViewWidget(plot)
            plot.addItem(Zc_Grid_Label)

        Vcyl_button = QPushButton("Plot Cylinder", self)
        Vcyl_button.setCheckable(False)
        Vcyl_button.toggle()
        Vcyl_button.clicked.connect(VCyl)
        
        layout.addWidget(Vcyl_button, 16, 1, 1, 1)

        def VSph():
            global pts
            
            n = 200
            theta_v = np.linspace(0,pts[2,1],n)
            theta = pts[2,1]

            phi_v = (np.linspace(0,pts[2,2],n))
            phi = pts[2,2]

            for i in range(n):
                x = np.array(pts[2,0]/4*np.sin(theta_v)*np.cos(theta))
                y = np.array(pts[2,0]/4*np.sin(theta)*np.sin(phi_v))
                z = pts[2,0]/4*np.cos(phi_v)
                
                x1 = np.array(pts[2,0]/4*np.sin(theta)*np.cos(phi_v))
                y1 = np.array(pts[2,0]/4*np.sin(theta)*np.sin(phi_v))
                z1 = np.zeros((0,200))

                pts_c = np.vstack([x,y,z]).transpose()
                plt2 = gl.GLLinePlotItem(pos=pts_c, color = (1,1,1,1), width=2)
                plot.addItem(plt2)

                pts_c1 = np.vstack([x1,y1,z1]).transpose()
                plt20 = gl.GLLinePlotItem(pos=pts_c1, color = (1,1,1,1), width=2)
                plot.addItem(plt20)
            
            rho_l = pts[2,0]
            theta_l = pts[2,1]
            phi_l = pts[2,2]

            R_Grid_Label = GLTextItem(X=rho_l/2*np.sin(theta_l)*np.cos(phi_l)+0.01, Y=rho_l/2*np.sin(theta_l)*np.sin(phi_l)+0.01, Z=rho_l*np.cos(phi_l)+0.05, text="R")
            
            Theta1_Grid_Label = GLTextItem(X=rho_l/4*np.sin(theta_l)*np.cos(phi_l), Y=rho_l/4*np.sin(theta_l)*np.sin(phi_l), Z=rho_l/4*np.cos(phi_l)+0.1, text="\u03C6")

            Phi_Grid_Label = GLTextItem(X=rho_l/4*np.sin(theta_l)*np.cos(phi_l), Y=rho_l/4*np.sin(theta_l)*np.sin(phi_l), Z=0, text="\u03B8")

          
            # Positive X
            R_Grid_Label.setGLViewWidget(plot)
            plot.addItem(R_Grid_Label)
            
            # Positive Y
            Theta1_Grid_Label.setGLViewWidget(plot)
            plot.addItem(Theta1_Grid_Label)

            # Positive Z
            Phi_Grid_Label.setGLViewWidget(plot)
            plot.addItem(Phi_Grid_Label)
               
            x,y,z = sp2cart(pts[2,0], pts[2,1], pts[2,2])


           
            z = pts[2,0]*np.cos(theta)
            phi_line_data = np.array(([0,0,0],[x,y,z])) 
            phi_line = gl.GLLinePlotItem(pos=phi_line_data, width=(4), antialias=True)
            plot.addItem(phi_line)

            new1 = np.array(([0,0,0],[x,y,0]))
            plt1 = gl.GLLinePlotItem(pos=new1, width=(4), antialias=True)
            plot.addItem(plt1)
        
            
            # sphere
            md1 = gl.MeshData.sphere(rows=20, cols=20, radius = (np.sqrt(x**2+y**2+z**2)**0.5))
            m6 = gl.GLMeshItem(meshdata=md1, color = (122,0,0,0.5), edgeColor=(0,0,0,1), smooth=True, drawFaces=False, drawEdges=True)
            plot.addItem(m6)

        Vsph_button = QPushButton("Plot Spherical", self)
        Vsph_button.setCheckable(False)
        Vsph_button.toggle()
        Vsph_button.clicked.connect(VSph)

        layout.addWidget(Vsph_button, 17, 1, 1, 1)

        pts_clear_button = QPushButton("Clear Conversions", self)
        pts_clear_button.setCheckable(False)
        pts_clear_button.toggle()
        pts_clear_button.clicked.connect(super_clear)
        layout.addWidget(pts_clear_button, 14, 1, 1, 1)


        clear_button = QPushButton("Clear Plot", self)
        clear_button.setCheckable(False)
        clear_button.toggle()
        clear_button.clicked.connect(super_clear)
        layout.addWidget(clear_button, 19, 1, 1, 1)

        ## Add widgets to the layout in their proper positions
        Logo = QLabel()
        Logo.setPixmap(QPixmap("/home/bigmeech/Documents/Personal/EE_307/1/UAH_blk.png").scaled(175,175,QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        Logo.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(Logo, 0, 0, 3, 3)

        layout.addWidget(X_int, 1, 1, 1, 1)   # text edit goes in top-left
        layout.addWidget(Y_int, 2, 1, 1, 1)   # text edit goes in middle-left
        layout.addWidget(Z_int, 3, 1, 1, 1)   # text edit goes in bottom-left

        layout.addWidget(R_int, 5, 1,1,1)   # text edit goes in top-left
        layout.addWidget(Phi_int, 6, 1, 1, 1)   # text edit goes in middle-left
        layout.addWidget(Zc_int, 7, 1, 1, 1)   # text edit goes in bottom-left

        layout.addWidget(Rho_int, 9, 1, 1, 1)   # text edit goes in top-left
        layout.addWidget(Theta_int, 10, 1, 1, 1)   # text edit goes in middle-left
        layout.addWidget(PhiS_int, 11, 1, 1, 1)  # text edit goes in bottom-left

        layout.addWidget(plot, 0, 3, 20, 90)  # plot goes on right side, spanning 3 rows       

        self.show()

def main():
    ## Always start by initializing Qt (only once per application)
    app = QtGui.QApplication(sys.argv)
    ex = MAIN()
    ## Define a top-level widget to hold everything
    sys.exit(app.exec_())
## Start the Qt event loop
if __name__ == '__main__':
    main()