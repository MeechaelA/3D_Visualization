from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import * 
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import sys
import os
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
        pts = np.empty(three, dtype = np.float)

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
                    
                    pts[2,0] = Rho
                    pts[2,1] = Theta
                    pts[2,2] = PhiS

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
                    print(pts[1,0], pts[1,1], pts[1,2])
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
            pts[0,0] = X_int
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
            pts[0,1] = Y_int
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
            pts[0,2] = Z_int
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
            pts[1,0] = R_int
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
            pts[1,1] = Phi_int
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
            pts[1,2] = Zc_int
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
            pts[2,0] = Rho_int
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
            pts[2,1] = Theta_int
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
            pts[2,2] = PhiS_int
            return pts[2,2] 

        PhiS_int.valueChanged[float].connect(PhiS_callback_int)
        PhiS_int.show()

        #Spherical Button
        sph_button = QPushButton("Convert Spherical", self)

        def Grids():# Grids
            
            gx = gl.GLGridItem(antialias = True)
            gx.rotate(90, 0, 1, 0)
            gx.translate(0, 0, 0)
            gx.setSpacing(x=1,y=1,z=1)
            plot.addItem(gx)
            
            gy = gl.GLGridItem(antialias = True)
            gy.rotate(90, 1, 0, 0)
            gy.setSpacing(x=1,y=1,z=1)
            plot.addItem(gy)
            
            gz = gl.GLGridItem(antialias = True)
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

            self.update()
            # Calculated Point
            #calc.setGLViewWidget(plot)
            
            #plot.addItem(calc)
        plot.setBackgroundColor(120,120, 120, 1)

        def plot_point():
            # Coordinate Pts.
            post = np.empty((1, 3))
            post[0] = (pts[0,0],pts[0,1],pts[0,2])

 
            co0 = gl.GLScatterPlotItem(pos=post,color = (255,0,0,1), size = 0.5, pxMode = False)
            plot.addItem(co0)
            
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
        Me = QLabel('Created by: Michael Angeles')
        layout.addWidget(Me, 1, 1, 1, 1)
        Me.setAlignment(QtCore.Qt.AlignTop)
        Instructor = QLabel('\nInstructed by: Dr. Adam Mehrabani')
        layout.addWidget(Instructor, 1, 1, 1, 1)
        Instructor.setAlignment(QtCore.Qt.AlignTop)

        ## Cartesian
        XLabel = QLabel('X:')
        layout.addWidget(XLabel, 2, 0)
        XLabel.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)

        YLabel = QLabel('Y:')
        layout.addWidget(YLabel, 3, 0)
        YLabel.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)

        ZLabel = QLabel('Z:')
        layout.addWidget(ZLabel, 4, 0)
        ZLabel.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)

        layout.addWidget(cart_button, 5, 1, 2,1, alignment = QtCore.Qt.AlignTop)

        #Cylindirical
        R_Label = QLabel("\u03C1:")
        layout.addWidget(R_Label, 6, 0)
        R_Label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)

        Phi_Label = QLabel('\u03C6:')
        layout.addWidget(Phi_Label, 7, 0)
        Phi_Label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)

        Zc_Label = QLabel('Z:')
        layout.addWidget(Zc_Label, 8, 0)
        Zc_Label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
        
        cyl_button.setCheckable(False)
        cyl_button.toggle()
        
        cyl_button.clicked.connect(cyl_conversion)
        layout.addWidget(cyl_button, 9, 1, 2, 1, alignment = QtCore.Qt.AlignTop)

        #Spherical 
        Rho_Label = QLabel('R:')
        layout.addWidget(Rho_Label, 10, 0)
        Rho_Label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)

        Theta_Label = QLabel('\u03B8:')
        layout.addWidget(Theta_Label, 11, 0)
        Theta_Label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)

        PhiS_Label = QLabel('\u03C6:')
        layout.addWidget(PhiS_Label, 12, 0)
        PhiS_Label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
        
        sph_button.setCheckable(False)
        sph_button.toggle()

        sph_button.clicked.connect(sph_conversion)
        layout.addWidget(sph_button, 13, 1, 2, 1, alignment = QtCore.Qt.AlignTop)

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
                plt = gl.GLLinePlotItem(pos=pts_c, color = (1, 1, 1, 1), width=2, antialias=True)
                plot.addItem(plt)

            x = (pts[1,0]*np.cos(pts[1,1]))
            y = (pts[1,0]*np.sin(pts[1,1]))
            z = pts[1,2]           

            theta_line_data = np.array(([0,0,0],[x,y,z])) 
            theta_line = gl.GLLinePlotItem(pos=theta_line_data, width=(2), antialias=True)
            plot.addItem(theta_line)
            
            new1 = np.array(([0,0,0],[x,y,0]))
            plt1 = gl.GLLinePlotItem(pos=new1, width=(2), antialias=True)
            plot.addItem(plt1)
            
            
            new2 = np.array(([x,y,0],[x,y,z])) 
            plt2 = gl.GLLinePlotItem(pos=new2, width=(2), antialias=True)
            plot.addItem(plt2)

            md = gl.MeshData.cylinder(rows=50, cols=100, radius=[np.sqrt(x**2 +y**2)-0.01, np.sqrt(x**2 +y**2)], length=z, offset = False)
            m5 = gl.GLMeshItem(meshdata=md, color = (0, 0, 0, 0.5), smooth=True, glOptions='translucent')
            plot.addItem(m5)

            rho = pts[1,0]
            phi = pts[1,1]
            z = pts[1,2]

            ax_vector = (0,0,0)
            ax0_vector = (np.cos(phi), np.sin(phi), 0)

            y0_vector = (0,0,0)
            y_vector = (-np.sin(phi), np.cos(phi), 0)
            
            z0_vector = (0,0,0)
            z_vector = (0,0,z)

            Arx = np.array([ax0_vector,ax_vector])
            Ary = np.array([y0_vector,y_vector])
            Arz = np.array([z0_vector,z_vector])

            Arho_line = gl.GLLinePlotItem(pos=Arx, color = (1,1,1,1), width=2, antialias=True)
            Arho_line.translate(rho*np.cos(phi), rho*np.sin(phi), z)
            plot.addItem(Arho_line)

            Aphi_line = gl.GLLinePlotItem(pos=Ary, color = (1,1,1,1), width=2, antialias=True)
            Aphi_line.translate(rho*np.cos(phi), rho*np.sin(phi), z)
            plot.addItem(Aphi_line)

            Az_line = gl.GLLinePlotItem(pos=Arz, color = (1,1,1,1), width=2, antialias=True)
            Az_line.translate(rho*np.cos(phi), rho*np.sin(phi), z)
            plot.addItem(Az_line)

            Arho_Grid_Label = GLTextItem(X = np.cos(phi), Y = np.sin(phi), Z = 0, text="a\u0302\u03C1")
            Arho_Grid_Label.translate(rho*np.cos(phi)+0.05, rho*np.sin(phi)+0.05, z+0.05)
            Arho_Grid_Label.setGLViewWidget(plot)
            plot.addItem(Arho_Grid_Label)
            
            Aphi_Grid_Label = GLTextItem(X= -np.sin(phi), Y=np.cos(phi), Z=0, text="a\u0302\u03C6")
            Aphi_Grid_Label.translate(rho*np.cos(phi)+0.05, rho*np.sin(phi)+0.05, z+0.05)
            Aphi_Grid_Label.setGLViewWidget(plot)
            plot.addItem(Aphi_Grid_Label)
          
            AZc_Grid_Label = GLTextItem(X=0, Y=0, Z=z, text="a\u0302z")
            AZc_Grid_Label.translate(rho*np.cos(phi)+0.05, rho*np.sin(phi)+0.05, z+0.05)
            AZc_Grid_Label.setGLViewWidget(plot)
            plot.addItem(AZc_Grid_Label)

            Rho_Grid_Label = GLTextItem(X=(pts[1,0]*np.cos(phi)+0.05), Y=(pts[1,0]*np.sin(phi)+0.05), Z=0, text="\u03C1")
            
            Phi_Grid_Label = GLTextItem(X=(pts[1,0]/2*np.cos(phi/2)+0.05), Y=(pts[1,0]/2*np.sin(phi/2)+0.05), Z=0, text="\u03C6")

            Zc_Grid_Label = GLTextItem(X=x+0.05, Y=y+0.05, Z=z/2, text="Z")

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
                x = np.array(pts[2,0]/4*np.sin(theta_v)*np.cos(phi))
                y = np.array(pts[2,0]/4*np.sin(theta)*np.sin(phi_v))
                z = pts[2,0]/4*np.cos(phi_v)
                
                x1 = np.array(pts[2,0]/4*np.sin(theta)*np.cos(phi_v))
                y1 = np.array(pts[2,0]/4*np.sin(theta)*np.sin(phi_v))
                z1 = np.zeros((0,200))

                pts_c = np.vstack([x,y,z]).transpose()
                plt2 = gl.GLLinePlotItem(pos=pts_c, color = (1,1,1,1), width=2, antialias=True)
                plot.addItem(plt2)

                pts_c1 = np.vstack([x1,y1,z1]).transpose()
                plt20 = gl.GLLinePlotItem(pos=pts_c1, color = (1,1,1,1), width=2, antialias=True)
                plot.addItem(plt20)
            
            rho_l = pts[2,0]
            theta_l = pts[2,1]
            phi_l = pts[2,2]

            R_Grid_Label = GLTextItem(X=rho_l*np.sin(theta_l-0.1)*np.cos(phi_l), Y = rho_l*np.sin(theta_l)*np.sin(phi_l)-0.1, Z = rho_l*np.cos(phi_l), text="R")
            
            Theta1_Grid_Label = GLTextItem(X=rho_l/4*np.sin(theta_l)*np.cos(phi_l), Y=rho_l/4*np.sin(theta_l)*np.sin(phi_l), Z=rho_l/4*np.cos(phi_l)+0.1, text="\u03B8")

            Phi_Grid_Label = GLTextItem(X=rho_l/4*np.sin(theta_l)*np.cos(phi_l/2), Y=rho_l/4*np.sin(theta_l/2)*np.sin(phi_l/2), Z=0, text="\u03C6")

            # Positive X
            R_Grid_Label.setGLViewWidget(plot)
            plot.addItem(R_Grid_Label)
            
            # Positive Y
            Theta1_Grid_Label.setGLViewWidget(plot)
            plot.addItem(Theta1_Grid_Label)

            # Positive Z
            Phi_Grid_Label.setGLViewWidget(plot)
            plot.addItem(Phi_Grid_Label)
               
            rho_l_AR = rho_l
            Ar_Grid_Label = GLTextItem(X = np.sin(theta)*np.cos(phi)*(1/2)+0.05,Y = np.sin(theta)*np.sin(phi)*(1/2)+0.05, Z = np.cos(theta)*(1/2)+0.05, text="a\u0302r")
            
            APhi_Grid_Label = GLTextItem(X = np.cos(theta)*np.cos(phi)*(1/2)+0.05, Y = np.cos(theta)*np.sin(phi)*(1/2)+0.05,Z = -np.sin(theta)*(1/2)+0.05, text="a\u0302\u03B8")

            ATheta1_Grid_Label = GLTextItem(X = -np.sin(phi)*(1/2) +0.05, Y = np.cos(phi)*(1/2)+0.05, Z = 0+0.05, text="a\u0302\u03C6")
            Ar_Grid_Label.translate(rho_l_AR*np.sin(theta_l)*np.cos(phi_l),rho_l*np.sin(theta_l)*np.sin(phi_l),rho_l*np.cos(phi_l))
            APhi_Grid_Label.translate(rho_l_AR*np.sin(theta_l)*np.cos(phi_l),rho_l*np.sin(theta_l)*np.sin(phi_l),rho_l*np.cos(phi_l))
            ATheta1_Grid_Label.translate(rho_l_AR*np.sin(theta_l)*np.cos(phi_l),rho_l*np.sin(theta_l)*np.sin(phi_l),rho_l*np.cos(phi_l))
          
            Ar_Grid_Label.setGLViewWidget(plot)
            plot.addItem(Ar_Grid_Label)
            
            x,y,z = sp2cart(pts[2,0], pts[2,1], pts[2,2])

            x0_vector = (0,0,0)
            x_vector = (rho_l_AR*np.sin(theta_l)*np.cos(phi_l),rho_l*np.sin(theta_l)*np.sin(phi_l),rho_l*np.cos(phi_l))
            rx = np.array([x0_vector,x_vector])
            r_line = gl.GLLinePlotItem(pos=rx, color = (1,1,1,1), width=2, antialias=True)
            plot.addItem(r_line)

            # Coordinate Pts.
            
            ax0_vector = (0,0,0)
            ax_vector = (np.sin(theta)*np.cos(phi)*(1/2),np.sin(theta)*np.sin(phi)*(1/2),np.cos(theta)*(1/2))

            y0_vector = (0,0,0)
            y_vector = (np.cos(theta)*np.cos(phi)*(1/2), np.cos(theta)*np.sin(phi)*(1/2),-np.sin(theta)*(1/2))
            
            z0_vector = (0,0,0)
            z_vector = (-np.sin(phi)*(1/2),np.cos(phi)*(1/2),0)

            Arx = np.array([ax0_vector,ax_vector])
            Ary = np.array([y0_vector,y_vector])
            Arz = np.array([z0_vector,z_vector])


            Ar_line = gl.GLLinePlotItem(pos=Arx, color = (1,1,1,1), width=2, antialias=True)
            Ar_line.translate(rho_l_AR*np.sin(theta_l)*np.cos(phi_l),rho_l*np.sin(theta_l)*np.sin(phi_l),rho_l*np.cos(phi_l))
            plot.addItem(Ar_line)

            Atheta_line = gl.GLLinePlotItem(pos=Ary, color = (1,1,1,1), width=2, antialias=True)
            Atheta_line.translate(rho_l_AR*np.sin(theta_l)*np.cos(phi_l),rho_l*np.sin(theta_l)*np.sin(phi_l),rho_l*np.cos(phi_l))
            plot.addItem(Atheta_line)

            Aphi_line = gl.GLLinePlotItem(pos=Arz, color = (1,1,1,1), width=2, antialias=True)
            Aphi_line.translate(rho_l_AR*np.sin(theta_l)*np.cos(phi_l),rho_l*np.sin(theta_l)*np.sin(phi_l),rho_l*np.cos(phi_l))
            plot.addItem(Aphi_line)

            # Positive Y
            ATheta1_Grid_Label.setGLViewWidget(plot)
            plot.addItem(ATheta1_Grid_Label)

            # Positive Z
            APhi_Grid_Label.setGLViewWidget(plot)
            plot.addItem(APhi_Grid_Label)

            x,y,z = sp2cart(pts[2,0], pts[2,1], pts[2,2])

            new1 = np.array(([0,0,0],[pts[0,0],pts[0,1],0]))
            plt1 = gl.GLLinePlotItem(pos=new1, width=(4), antialias=True)
            plot.addItem(plt1)
        
            # sphere
            md1 = gl.MeshData.sphere(rows=20, cols=20, radius = pts[2,0])
            m6 = gl.GLMeshItem(meshdata=md1, color = (120, 120, 120,0.2), edgeColor=(0,0,0,1), smooth=True, drawFaces=False , drawEdges=True)
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

        def clear_plot():
            plot.removeItem(co0)
            plot.removeItem(co)
            plot.removeItem(co1)
            plot.removeItem(co2)
            plot.removeItem(plt)
            plot.removeItem(theta_line)
            plot.removeItem(plt1)
            plot.removeItem(plt2)
            plot.removeItem(m5)
            plot.removeItem(Rho_Grid_Label)
            plot.removeItem(Phi_Grid_Label)
            plot.removeItem(Zc_Grid_Label)
            plot.removeItem(plt2)
            plot.removeItem(plt20)
            plot.removeItem(R_Grid_Label)
            plot.removeItem(Theta1_Grid_Label)
            plot.removeItem(Phi_Grid_Label)
            plot.removeItem(Ar_Grid_Label)
            plot.removeItem(ATheta1_Grid_Label)
            plot.removeItem(APhi_Grid_Label)
            plot.removeItem(phi_line)
            plot.removeItem(plt1)
            plot.removeItem(m6)

        clear_button = QPushButton("Clear Plot", self)
        clear_button.setCheckable(False)
        clear_button.toggle()
        clear_button.clicked.connect(clear_plot)
        layout.addWidget(clear_button, 19, 1, 1, 1)

        grid_button = QPushButton("Show Grid", self)
        grid_button.setCheckable(False)
        grid_button.toggle()
        grid_button.clicked.connect(Grids)
        layout.addWidget(grid_button, 18, 1, 1, 1)
        cwd = os.getcwd()
        ## Add widgets to the layout in their proper positions
        Logo = QLabel()
        print(cwd)
        Logo.setPixmap(QPixmap(cwd +"/UAH_blk.png").scaled(175,175,QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        Logo.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(Logo, 0, 0, 1, 3)

        layout.addWidget(X_int, 2, 1, 1, 1)   # text edit goes in top-left
        layout.addWidget(Y_int, 3, 1, 1, 1)   # text edit goes in middle-left
        layout.addWidget(Z_int, 4, 1, 1, 1)   # text edit goes in bottom-left

        layout.addWidget(R_int, 6, 1,1,1)   # text edit goes in top-left
        layout.addWidget(Phi_int, 7, 1, 1, 1)   # text edit goes in middle-left
        layout.addWidget(Zc_int, 8, 1, 1, 1)   # text edit goes in bottom-left

        layout.addWidget(Rho_int, 10, 1, 1, 1)   # text edit goes in top-left
        layout.addWidget(Theta_int, 11, 1, 1, 1)   # text edit goes in middle-left
        layout.addWidget(PhiS_int, 12, 1, 1, 1)  # text edit goes in bottom-left

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
