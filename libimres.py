#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 07:31:59 2020

@author: thomas
"""

import libconstants as const
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from itertools import cycle
import time

"""
System parameters and constants
"""

def arr_sum(a, i,bin_size):
    
    b = np.zeros([bin_size])
    
    for j in range(a.size):
        
        b[ i[j] ] += a[j]
        
    return b
    
    
    
# =============================================================================
# Estimation of the pdf of samples
# =============================================================================
def estimate_pdf(p, bins = 20):
    h = np.histogram(p, bins = bins)
    y = h[0]
    x = (h[1][1:] + h[1][0 : y.size] )/2
    Dx = x[1] - x[0]
    y = y / Dx / p.size
    return x, y

# =============================================================================
# Generate samples with arbritary distribution
# invcdf is the inverse of the cumulative probability density function of the
# desired distribution
# =============================================================================
def rand_cdf(no_samples, invcdf):
    
    x = np.random.rand(no_samples)
    y = invcdf(x)
    return y

# =============================================================================
# inverse CDF function for the cos(t) distribution
# =============================================================================
def invcdf_cos(x):
    return np.arcsin( x )

# =============================================================================
# Generate samples with cos(t) distribution. To be used in later implementation 
# where higher order lambertian transmitters can be used
# =============================================================================
def rand_cos(no_samples):
    return rand_cdf(no_samples, invcdf_cos)

# =============================================================================
# Plot a vector originating at rstart and ending at rend
# =============================================================================
def plot_vector(rstart, rend, color = 'blue'):
    plt.plot( [rstart[0], rend[0]], [rstart[1], rend[1]], [rstart[2], rend[2]],'-', color = color)
    plt.plot( [rend[0]], [rend[1]], [rend[2]],'o', color = color )
    
# =============================================================================
# Visualize a set of vectors 
# =============================================================================
def visualize_vectors( vectors, rstart = np.array([0.0, 0.0, 0.0]), color = 'blue' ):
    
    no_of_vectors, _ = vectors.shape
    fig = plt.gcf()
    ax = fig.gca(projection='3d')

    for i in range(no_of_vectors):
        rend = vectors[ i ]
        plot_vector(rstart, rend)

# =============================================================================
# Direction of reflected wave assuming specular (mirror-like) reflection
# =============================================================================  
def direction_of_reflected(k,n):
    
    l = np.dot(k,n)
    return - 2.0 * l * n + k

# =============================================================================
# Rotation matrix obtained by rotating the 3D coordinate around the x-axis with
# an angle equal to t   
# =============================================================================
def Rx(t):
    return np.array([[1,         0,          0],
                     [0, np.cos(t), -np.sin(t)],
                     [0, np.sin(t), np.cos(t) ]])

# =============================================================================
# Rotation matrix obtained by rotating the 3D coordinate around the y-axis with
# an angle equal to t   
# =============================================================================    
def Ry(t):
    return np.array([[np.cos(t), 0,  np.sin(t)],
                     [0,         1,          0],
                     [-np.sin(t),0, np.cos(t) ]])

# =============================================================================
# Rotation matrix obtained by rotating the 3D coordinate around the z-axis with
# an angle equal to t   
# =============================================================================    
def Rz(t):
    return np.array([[np.cos(t), -np.sin(t), 0],
                     [np.sin(t),  np.cos(t), 0],
                     [        0,          0, 1]])
# =============================================================================
# rotate the vector r around the x-axis with an agle equal to t                    
# =============================================================================
def rotate_x(r,t):
    return np.matmul(Rx(t), np.transpose(r))

# =============================================================================
# rotate the vector r around the y-axis with an agle equal to t                    
# =============================================================================
def rotate_y(r,t):
    return np.matmul(Ry(t), np.transpose(r))

# =============================================================================
# rotate the vector r around the z-axis with an agle equal to t                    
# =============================================================================
def rotate_z(r,t):
    return np.matmul(Rz(t), np.transpose(r))

# =============================================================================
# skew symmetric matrix obtained from vector v
# =============================================================================
def skew_symmetric(v):
    
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

# =============================================================================
# rotation matrix so that vector a coincides with vector b    
# =============================================================================
def rotation_matrix(a,b):
    
    a = a / np.linalg.norm( a )
    b = b / np.linalg.norm( b )
    
    v = np.cross(a,b)
    c = np.inner(a,b)
    
    V = skew_symmetric(v)
    
    if c==-1.0:
        M = - np.eye(3)
    else:
        M = np.eye(3) + V + np.linalg.matrix_power(V,2) / (1+c)
    
    return M

# =============================================================================
# generation of random orientation vectors required for non-specular lambertian
# reflection (mode = 1)
# =============================================================================
def generate_orientation_vectors( no_vectors, 
                                  orientation = np.array([0.0, 0.0, 1.0]) ):
    
    R = rotation_matrix( np.array([0.0, 0.0, 1.0]), orientation)
    orientation = orientation / np.linalg.norm( orientation )    
    theta = np.pi/2.0 * np.random.rand( no_vectors )
    phi = 2.0 * np.pi * np.random.rand( no_vectors ) 
    vectors = np.zeros( [no_vectors, 3] )
    vectors[:,0] = np.sin(theta) * np.sin(phi)
    vectors[:,1] = np.sin(theta) * np.cos(phi)    
    vectors[:,2] = np.cos(theta)
    
    for i in range(vectors.shape[0]):
        vectors[i,:] = np.matmul( R, vectors[i,:] )
        
    return vectors

# =============================================================================
# generate single direction of reflected wavea assuming non-specular lambertian 
# reflection (mode = 1)
# =============================================================================
def direction_of_diffusive( n ):
    R = rotation_matrix( np.array([0.0, 0.0, 1.0]), n )
    theta = np.pi / 2.0 * np.random.rand( 1 )     
    phi = 2.0 * np.pi * np.random.rand( 1 ) 
    vector = np.zeros( [3] )
    vector[0] = np.sin(theta) * np.sin(phi)
    vector[1] = np.sin(theta) * np.cos(phi)    
    vector[2] = np.cos(theta)
    vector = np.matmul(R, vector )
    return vector

# =============================================================================
# find right handed normal to vectors a and b
# =============================================================================
def normal_to_vectors(a,b):
    a = a / np.linalg.norm( a )
    b = b / np.linalg.norm( b )
    
    v = np.cross(a,b)
    v = v / np.linalg.norm(v)
    return v

# =============================================================================
# calculate function over a meshgrid defined by axes x and y
# =============================================================================
def expandfun(x,y,fun):
    
    xx,yy=np.meshgrid(x,y)
    return fun(xx,yy)

# =============================================================================
# calculate a function over a 3D-like meshgrid defined by the coordinate of the
# 3D vectors v1 and v2
# =============================================================================
def expandfunvec3(v1,v2,fun):
    
    x1=v1[:,0]
    y1=v1[:,1]
    z1=v1[:,2]
    
    x2=v2[:,0]
    y2=v2[:,1]
    z2=v2[:,2]
    
    [xx1,xx2]=np.meshgrid(x1,x2)
    [yy1,yy2]=np.meshgrid(y1,y2)
    [zz1,zz2]=np.meshgrid(z1,z2)
    
    f=fun(xx1,yy1,zz1,xx2,yy2,zz2)
    
    return f

# =============================================================================
# calculate a function over a 3D-like meshgrid defined by the coordinate of the
# 3D vectors v1, v2 and v3,v4
# =============================================================================
def expandfunvec32(v1,v2,v3,v4,fun):
    
    x1=v1[:,0]
    y1=v1[:,1]
    z1=v1[:,2]
    
    x2=v2[:,0]
    y2=v2[:,1]
    z2=v2[:,2]
    
    x3=v3[:,0]
    y3=v3[:,1]
    z3=v3[:,2]
    
    x4=v4[:,0]
    y4=v4[:,1]
    z4=v4[:,2]
    
    [xx1,xx2]=np.meshgrid(x1,x2)
    [yy1,yy2]=np.meshgrid(y1,y2)
    [zz1,zz2]=np.meshgrid(z1,z2)
        
    [xx3,xx4]=np.meshgrid(x3,x4)
    [yy3,yy4]=np.meshgrid(y3,y4)
    [zz3,zz4]=np.meshgrid(z3,z4)
    
    f=fun(xx1,yy1,zz1,xx2,yy2,zz2,xx3,yy3,zz3,xx4,yy4,zz4)
    
    return f       

def calc_distances(rTs,rRs):
    
    funRR = lambda rRx,rRy,rRz,rTx,rTy,rTz, : np.sqrt( (rRx-rTx)**2+(rRy-rTy)**2+(rRz-rTz)**2)
    RR=expandfunvec3(rRs,rTs,funRR)
    return(RR)

    
# =============================================================================
# calculate line-of-sight (LOS) component obtained by pairs of N transmitters and 
# M receivers. 
# rTs and rRs are arrays (N x 3) and (M x 3) containing the positions of the 
# transmitters and receivers respectively,
# nTs and nRs are arrays (N x 3) and (M x 3) containing the orientations of the 
# transmitters and receivers respectively,
# ARs, FOVs are an arrays with M size containing the areas and 
# the field of views of the receivers 
# ns is an array of N size containing the mode orders of the transmitters
# etas is an array of M size containing the efficiency of the receivers
# =============================================================================
def calcHDC(rTs,rRs,nTs,nRs,ARs,FOVs,ns,etas):
    
    [Nt,nnn]=rTs.shape
    [Nr,nnn]=rRs.shape
        
    RR = calc_distances(rTs,rRs)
    
    u = np.where(RR != 0)
  
    funcostheta = lambda rRx,rRy,rRz,rTx,rTy,rTz,nRx,nRy,nRz,nTx,nTy,nTz : \
                (rTx-rRx)*nRx+(rTy-rRy)*nRy+(rTz-rRz)*nRz 
    
    expand_cos = expandfunvec32(rRs,rTs,nRs,nTs,funcostheta) 
    costheta = np.zeros( RR.shape )
    costheta[u] = expand_cos[u] / RR[u]
    
    funcosphi = lambda rRx,rRy,rRz,rTx,rTy,rTz,nRx,nRy,nRz,nTx,nTy,nTz : \
                (rRx-rTx)*nTx+(rRy-rTy)*nTy+(rRz-rTz)*nTz 
                
    cosphi = np.zeros( RR.shape )
    expand_cosphi = expandfunvec32(rRs,rTs,nRs,nTs,funcosphi)
    cosphi[u] =  expand_cosphi[u] / RR[u]
    
    etass=np.tile(etas,[Nt,1])
    ARss = np.tile(ARs,[Nt,1])
    FOVss = np.tile(FOVs,[Nt,1])
    nss=np.tile(ns.reshape(Nt,1),[1,Nr])
    
    Itheta = np.less_equal(np.arccos(costheta),FOVss).astype(float) 
    Iphi = np.less_equal(np.arccos(cosphi),np.pi/2.0).astype(float) 
    
    HDC = np.zeros( RR.shape )
    HDC[u] = etass[u] * ( nss[u] + 1.0 ) / 2.0 / np.pi / RR[u] ** 2.0 * ARss[u]
    HDC = HDC * costheta * cosphi ** nss * Itheta * Iphi
    return HDC

# =============================================================================
# calculate line-of-sight (LOS) power components obtained by pairs of N 
# transmitters and M receivers. 
# rTs and rRs are arrays (N x 3) and (M x 3) containing the positions of the 
# transmitters and receivers respectively,
# nTs and nRs are arrays (N x 3) and (M x 3) containing the orientations of the 
# transmitters and receivers respectively,
# ARs, FOVs are an arrays with M size containing the areas and 
# the field of views of the receivers 
# ns is an array of N size containing the mode orders of the transmitters
# etas is an array of M size containing the efficiency of the receivers
# =============================================================================
def calcPcontr(rTs,rRs,nTs,nRs,ARs,FOVs,PTs,ns,etas):
    PTs = PTs.reshape([PTs.size,1])    
    HDC = calcHDC(rTs,rRs,nTs,nRs,ARs,FOVs,ns,etas)
    return HDC * PTs 


# =============================================================================
# calculate line-of-sight (LOS) powers obtained on M receivers by N 
# transmitters. 
# rTs and rRs are arrays (N x 3) and (M x 3) containing the positions of the 
# transmitters and receivers respectively,
# nTs and nRs are arrays (N x 3) and (M x 3) containing the orientations of the 
# transmitters and receivers respectively,
# ARs, FOVs are an arrays with M size containing the areas and 
# the field of views of the receivers 
# ns is an array of N size containing the mode orders of the transmitters
# etas is an array of M size containing the efficiency of the receivers
# =============================================================================
def calcP(rTs,rRs,nTs,nRs,ARs,FOVs,PTs,ns,etas):
    PTs = PTs.reshape([PTs.size,1])
    HDC = calcHDC(rTs,rRs,nTs,nRs,ARs,FOVs,ns,etas)
    return np.sum(PTs * HDC, axis = 0)

# =============================================================================
# if a is scalar tile it to an array of given shape. Otherwise just return a    
# =============================================================================
def to_array(a, shape):
    
    if np.isscalar(a):
        return a * np.ones( [shape[0]] )
    else:
        return a

# =============================================================================
# if a is a 3D-vector tile to an array of given shape. Otherwise return a
# =============================================================================
def to_array3(a, shape):
    
    if len(a.shape) == 1:
        return np.tile(a, [ shape[0],1 ] )
    else:
        return a
    

# =============================================================================
# Ray class for representing an optical ray and its path
# =============================================================================
class ray:
    
    def __init__(self,
                 position = np.array([0.0,0.0,0.0,]),
                 time = 0.0,
                 orientation = np.array([0.0,0.0,1.0]),
                 power = 1.0):
        self.position = position
        self.time = time
        self.orientation = orientation / np.linalg.norm( orientation )
        self.trajectory = [(position, orientation, time, power)]
        self.power = power
        self.currently_on = None

# =============================================================================
#   Calculate distance of ray and a given point in the 3D axis
# =============================================================================       
    def distance_from_point(self,point):
                
        return np.linalg.norm( np.cross( self.position - point, self.orientation ) )
    
# =============================================================================
# Determine whether the ray intersects the surfaces surf
#
# USE WITH CAUTION THIS HAS NOT BEEN TESTED!!!
#        
# =============================================================================
    def instersects_surface(self, surf):
        
        for i, point in enumerate(surf.points):
            d = self.distance_from_point(point) 
            if (d <= surf.D / 2.0):
                if np.dot( point - self.position, self.orientation) > 0:
                   return point, d            
        
        return None

# =============================================================================
# Perform a single bounce of the ray inside the collection of surfaces R. 
# reflection_type can be 'specular' or 'diffusive' implying specular or 
# lambertian reflection (mode = 1) at the incident surface. This is determined
# by the reflection_type attribute of the surfaces
# =============================================================================    
    def bounce(self, R):
        
        minimum_distance = np.Inf
        p_closest = None
        n_closest = None
        p_original = self.position
        
        for i,key in enumerate(R.surfaces):
            surf = R.surfaces[ key ]
            
            if surf == self.currently_on:
                pass
            else:
                result = surf.ray_intersection( self )
                if result is not None:
                    (p, d, index) = result
                    if d < minimum_distance:
                        p_closest = p
                        n_closest = surf.normal[index]
                        r_closest = surf.reflectivity[index]
                        surf_closest = surf
                        
        
        if n_closest is not None:
            reflection_type = surf.reflection_type
            if reflection_type == 'specular':
                self.orientation = direction_of_reflected( self.orientation, n_closest )
            elif reflection_type == 'lambertian':
                self.orientation = direction_of_diffusive( n_closest )
                
            self.time = self.time + np.linalg.norm(p_original - p_closest)
            self.power = self.power * r_closest    
            self.trajectory.append( (p_closest, self.orientation, self.time, self.power )  )
            self.currently_on = surf_closest
            self.position = p_closest
            
            return surf_closest, p_closest, n_closest
        else:
            return None, None, None
            
    def visualize_trajectory(self, show_labels = True):
        ax = plt.gca(projection = '3d')


        for i, tr in enumerate(self.trajectory):
            
            (p1,_,t,power1) = self.trajectory[ i ]                
            if (i < len(self.trajectory)-1 ):
                (p2,_,_,power2) = self.trajectory[ i+1 ]
                print(p1,'[',power1,']','-->',p2,' [',power2,']')
                plt.plot( [p2[0], p1[0]], [p2[1], p1[1]], [p2[2], p1[2]],'-' ) 
            plt.plot( [p1[0]], [p1[1]], [p1[2]], marker = 'o', color = 'red')
            if show_labels:
                time_label = '%2.1f (%2.2f)' % (t,power1)
                ax.text(p1[0], p1[1], p1[2], time_label, zdir=None)
                
                
            #plt.plot( [p2[0], p2[1], p2[2]], 'o', color='r')
                
            
class surface:
    
    def __init__(self,points = None, 
                 normal = np.array([0.0,0.0,1.0]), 
                 power = 0.0, 
                 name = '',
                 received_power = 0.0, 
                 A = 0.0, 
                 ns = 1.0, 
                 FOV = np.pi/2.0, 
                 eta = 1.0,
                 reflectivity = 0.8,
                 reflection_type = 'lambertian'):

        self.points = points             
        self.name = name
            
        if points is None:
            self.normal = None
            self.power = None
            self.received_power = None
            self.A = None
            self.ns = None
            self.FOV = None
            self.eta = None
            self.reflectivity = None
        
        else:                  
            self.normal = to_array3(normal, points.shape)       
            self.power = to_array(power, points.shape)
            self.received_power = to_array(power, points.shape)
            self.A = to_array(A, points.shape)
            self.ns = to_array(ns, points.shape)
            self.FOV = to_array(FOV, points.shape)
            self.eta = to_array(eta, points.shape)
            self.reflectivity = to_array(reflectivity, points.shape)
            
        self.rotations = []
        self.reflection_type = reflection_type
    
    def number_of_points(self):
        if self.points is not None:
            return self.points.shape[0]
        else:
            return 0
        
    def visualize(self,color = 'r', marker = 'o', legend = None, alpha = 1.0, new_figure = False):
        
        if new_figure:
            fig = plt.figure()
        else:
            fig = plt.gcf()
        
        ax=fig.gca(projection='3d')
        
        ax.scatter3D(self.points[:,0],
                     self.points[:,1],
                     self.points[:,2], 
                     color = color,marker = marker, label = legend, alpha = alpha)
        
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        if legend is not None:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.5)
        
        plt.tight_layout()
        
    def rotate(self, axis = 'x', angle = np.pi/4):
        
        rotation = {'axis' : axis,
                    'angle' : angle}
        
        self.rotations.append(rotation)
        
        if axis == 'x' :
            R = lambda r : rotate_x(r,angle)
        elif axis == 'y' :            
            R = lambda r : rotate_y(r,angle)
        elif axis =='z' :
            R = lambda r : rotate_z(r,angle)
            
        for i in range( 0, self.number_of_points() ):
            
            self.points[i, :] = R( self.points[i, :] )
            self.normal[i,:] = R( self.normal[i, :] )
            
            
    def displace(self, r0 = np.array([0.0, 0.0, 0.0]) ):
        
        for i in range(0,self.number_of_points()):
            self.points[i,:] = self.points[i, :] + r0
    
    def set_normal(self, n):
         self.normal = to_array3(n, self.points.shape)
            
    def set_area(self, A):
        self.A = to_array(A, self.points.shape)
    
    def set_FOV(self, FOV):
        self.FOV = to_array(FOV, self.points.shape)
    
    def set_eta(self, eta):
        self.eta = to_array(eta, self.points.shape)
    
    def set_ns(self, ns):
        self.ns = to_array(ns, self.points.shape)
    
    def set_power(self, power):
        
        if np.isscalar(power) :
            self.power = to_array(power, self.points.shape)
            
    def set_reflectivity(self, reflectivity):
        self.reflectivity = to_array(reflectivity, self.points.shape)    
        
    def set_received_power(self, power):
        self.received_power = to_array(power, self.points.shape)
    
    def add_point(self, 
                  point = np.array([0.0,0.0,0.0]), 
                  normal = np.array([0.0,0.0,1.0]),
                  power = 0.0):

        if self.points is None:
            self.points = np.array([point])
            self.normal = np.array([normal])
            self.power = np.array([power])
        else:
            self.points = np.append(self.points, [point], axis = 0)
            self.normal = np.append(self.normal, [normal], axis = 0)
            self.power = np.append(self.power, [power] )
    
    def transform_points(self, rotation_matrix = np.eye(3), 
                               displacement = np.zeros([3]), 
                               rotation_first = True):
        
        if rotation_first:
            transformation = lambda r: np.matmul( rotation_matrix, np.transpose(r) ) + displacement
        else:
            transformation = lambda r: np.matmul( rotation_matrix, np.transpose(r) + displacement ) 
        
        for i, point in enumerate(self.points):
            self.points[i,:] = transformation( point )
            self.normal[i,:] = np.matmul( rotation_matrix, self.normal[i,:] )   
    
    def calc_power_from(self, S, update_surface = True):
        received_power = calcP(
                S.points, self.points, S.normal, self.normal, self.A, 
                self.FOV, S.power , S.ns, self.eta)
    
        if update_surface:
            self.received_power = received_power
        else:
            return received_power
    
    def power_contributions_from(self,source_s):
        
        return calcPcontr(source_s.points, self.points, 
                          source_s.normal, self.normal,
                          self.A, self.FOV, 
                          source_s.power, source_s.ns, self.eta)
# =============================================================================
# Determine whether a ray intersects the surface
# Note: this uses exhaustive search and is slow.    
# =============================================================================    
    def ray_intersection(self,ray_obj):
        
        for i, point in enumerate(self.points):
            d = ray_obj.distance_from_point(point)
            if (d <= self.D * 0.707 ):
                if np.dot( point - ray_obj.position, ray_obj.orientation) > 0:                   
                   return point, d, i           
        return None

# =============================================================================
# Calculate distance of all surface points from a given point    
# =============================================================================
    def distance_from_point(self,point = np.array([0.0, 0.0, 0.0]) ):
         
        d = (self.points[:,0] - point[0]) ** 2.0 + \
            (self.points[:,1] - point[1]) ** 2.0 + \
            (self.points[:,2] - point[2]) ** 2.0
            
        return np.sqrt(d)
    
    def distance_from_surface(self,S):
        
        rTs = S.points
        rRs = self.points
        return calc_distances(rTs,rRs)
        
        

# =============================================================================
# A surface comprising of just a single point    
# =============================================================================    
class single_point(surface):
    
    def __init__(self,point = np.array([0.0, 0.0, 0.0]), normal = np.array([0.0, 0.0, -1.0]), *args,**kwargs):

        
        points = np.array([[ point[0], point[1], point[2] ]])
        normal = np.array([[ normal[0], normal[1], normal[2] ]])
        
        super().__init__(*args, points = points, normal = normal, **kwargs)
        

# =============================================================================
# Surface on the xy plane described by functions on paremeter on t
# =============================================================================        
class parametric_surface(surface):
   
# =============================================================================
# genereation of a parametric surface
# parameter_range is a numpy array of shape [N,2] where N is the numbero fo points
# containing the values of the two parameters,
# power_function is the power profile on the surface
# generating_function is the generating function of the surface
# gradient is a function describing the normal on the surface
# =============================================================================     
    def __init__(self, *args, 
                 generating_function = lambda self, t : np.array([ t[0], t[1], 0]),
                 power_function = lambda self, t : 0.0, 
                 gradient = lambda self, t : np.array([ 0.0, 0.0, 1.0]),
                 parameter_range = [np.arange(0,1,0.1), np.arange(0,1,0.1)],
                 **kwargs):
        
        self.generating_function = generating_function  # equation describing the surface
        self.power_function = power_function
        self.gradient = gradient
        self.parameter_range = parameter_range
        self.tmin0 = np.min(self.parameter_range[0])
        self.tmax0 = np.max(self.parameter_range[0])
        self.tmin1 = np.min(self.parameter_range[1])
        self.tmax1 = np.max(self.parameter_range[1])
        
        super().__init__(*args,**kwargs)
        
        for t0 in self.parameter_range[0]:
            for t1 in self.parameter_range[1]:
                
                t = np.array([t0, t1])
                point = self.generating_function(self,t)
                n = self.gradient(self,t)
                p = self.power_function(self,t)
                
                if self.points is None:
                    self.points = np.array([point])
                    self.normal = np.array([n])
                    self.power = np.array([p])
                    self.parameters_t = np.array([t])
                else:
                    self.points = np.append(self.points, [point], axis = 0)
                    self.normal = np.append(self.normal, [n], axis = 0)
                    self.power = np.append(self.power, [p] )
                    self.parameters_t = np.append(self.parameters_t, [t], axis = 0)

# =============================================================================
# visualize a distribution on the parametric surface    
# =============================================================================    
    def visualize_on_surface(self, distribution, new_figure = False):
        
        if new_figure:
            fig = plt.figure()
        else:
            fig = plt.gcf()
        
        t0 = self.parameter_range[0]
        t1 = self.parameter_range[1]
        
        dist = np.reshape(distribution,[t0.size, t1.size])
        fig = plt.figure()
        plt.pcolor(t0, t1, np.transpose(dist) )
        plt.colorbar()
        plt.axis('equal')                               

# =============================================================================
# Visualize surface received power on parametric surface
# =============================================================================        
    def visualize_received_power(self, new_figure = False):
        
        self.visualize_on_surface( self.received_power )

# =============================================================================
# Rectangular surface#         
# =============================================================================        
class rectangle(parametric_surface):
    
    def __init__(self, *args,                  
                 D = 0.05,                
                 power_per_element = 0.0,  
                 ns = 1.0,
                 eta = 1.0,
                 FOV = np.pi / 2.0, 
                 vectors = None,
                 normal = None,
                 origin = np.array([0.0, 0.0, 0.0]),
                 **kwargs):
        
        A = D ** 2.0
        
        if vectors is not None:
            
            if normal is None:
                normal = normal_to_vectors(
                        vectors[0,:], vectors[1,:])
            
            rA = origin
            rB = origin + vectors[0,:]
            rC = origin + vectors[1,:]
            
            Rz = rotation_matrix(normal,const.orient['+z'])
            
            rAr = np.matmul(Rz,rA)
            rBr = np.matmul(Rz,rB)
            rCr = np.matmul(Rz,rC)
            
            Rx = rotation_matrix(rBr-rAr,const.orient['+x'])
            
            rAr = np.matmul(Rx,rAr)
            rBr = np.matmul(Rx,rBr)
            rCr = np.matmul(Rx,rCr)
            
            rAt = rBr - rAr
            rBt = rCr - rAr
            
            xmin = np.min( np.array( [rAt[0], rBt[0] ] ) )
            ymin = np.min( np.array( [rAt[1], rBt[1] ] ) )
            xmax = np.max( np.array( [rAt[0], rBt[0] ] ) )
            ymax = np.max( np.array( [rAt[1], rBt[1] ] ) )
            
#            x = np.arange(xmin, xmax + D, D)
#            y = np.arange(ymin, ymax + D, D)

            x = np.arange(xmin + D/2.0, xmax, D)
            y = np.arange(ymin + D/2.0, ymax, D)
            
            parameter_range = [x, y]
            super().__init__(*args, parameter_range = parameter_range, A = A, **kwargs)
            R = np.matmul(Rx,Rz)            
            self.transform_points( displacement = rAr, rotation_matrix = np.linalg.inv( R ), 
                                   rotation_first = False )
            self.D = D
            self.set_area( A )
            self.set_power( power_per_element )
            self.set_received_power( 0 )
            self.set_ns( ns )
            self.set_eta( eta )
            self.set_FOV( FOV )
            self.set_normal( normal )
            self.rot_matrix = R
            self.rot_matrix_inv = np.linalg.inv( R )
            self.displacement = -rAr

# =============================================================================
# Determine whether the ray intersects the surfaces surf
# =============================================================================      
    def ray_intersection(self,ray_obj,exhaustive_search = False):
         
         if exhaustive_search:
             return super().ray_intersection(ray_obj)
         
         else:    
             r = np.matmul(self.rot_matrix, ray_obj.position )        
             r = r + self.displacement
             t = np.matmul(self.rot_matrix, ray_obj.orientation )
             if t[2] == 0.0:
                 return None
             else:            
                 l = - r[2] / t[2]
                 ri = r + l * t
                 if  (self.tmin0-self.D/2 < ri[0] < self.tmax0 + self.D/2) and \
                     (self.tmin1-self.D/2 < ri[1] < self.tmax1 + self.D/2): 
                     i = np.round( (ri[0] - self.tmin0)/self.D )
                     j = np.round( (ri[1] - self.tmin1)/self.D )
                     
                     rappx = np.array([self.tmin0 + i*self.D, self.tmin1 + j*self.D, 0.0])
                     int_point = rappx - self.displacement
                     int_point = np.matmul(self.rot_matrix_inv, int_point )
                     index = int(i * self.parameter_range[1].shape[0] + j)
                     
                     return int_point, np.linalg.norm(int_point - ray_obj.position), index  
                 else:
                     return None
            
# =============================================================================
# Room class containing all the surfaces           
# =============================================================================           
class room:

    def corners( wall_name, height = None, width = None, length = None ):
        
        if wall_name == 'ceiling':
            corners = np.array([[0.0, 0.0, height],
                                [0.0, width, height],
                                [length, 0.0, height]])
        
        elif wall_name == 'floor':
            corners = np.array([[0.0, 0.0, 0.0],                                
                                [length, 0.0, 0.0],
                                [0.0, width, 0.0]])
    
        elif wall_name == 'north':
            corners = np.array([[0.0, width, 0.0 ],                                
                                [length, width, 0.0],
                                [0.0, width, height ]])
        elif wall_name == 'south':
            corners = np.array([[0.0, 0.0, 0.0 ],
                                [0.0, 0.0, height ],
                                [length, 0.0, 0.0]])
    
        elif wall_name == 'east':
            corners = np.array([[length, 0.0, 0.0 ],
                                [length, 0.0, height ],
                                [length, width, 0.0]])
    
        elif wall_name == 'west':
            corners = np.array([[0.0, 0.0, 0.0 ],                                
                                [0.0, width, 0.0],
                                [0.0, 0.0, height] ])
    
        return corners
    
    def __init__(self, corners = None, names = None, D = 0.01):
        
        self.surfaces = {}
        no_corners, _ = corners.shape               
        
        for i in range(0, no_corners, 3):
            
            A = corners[i, :]
            B = corners[i+1, :]
            C = corners[i+2, :]
            
            vectors = np.array([ B-A, C-A ])
    
            if names is None:
                name = 'wall' + str( i + 1 )
            else:
                name = names.pop( 0 )

            self.surfaces[ name ] = rectangle( origin = A, vectors = vectors, D = D, name = name)
            
    def visualize_surfaces(self, alpha = 1.0, colors = None, markers = None):
            
        if markers is None:
            markers = ['o']
        
        if colors is None:
            colors = ['r']
        
        colors = cycle(colors)
        markers = cycle(markers)
        
        for key in self.surfaces:
            marker = next(markers)
            color = next(colors)
            self.surfaces[ key ].visualize(alpha = alpha, legend = key, marker = marker, color = color)
            
# =============================================================================
#     concatenate all room surface elements to a single one
# =============================================================================
    def concatenate_surfaces(self):
        
        S = surface()
        for key in self.surfaces:
            surf = self.surfaces[key]
            if S.points is None:
                S.points = surf.points
                S.normal = surf.normal
                S.A = surf.A
                S.ns = surf.ns
                S.reflectivity = surf.reflectivity
                S.eta = surf.eta
                S.FOV = surf.FOV
                
            else:
                S.points = np.concatenate( (S.points, surf.points ), axis = 0 )
                S.normal = np.concatenate( (S.normal, surf.normal ), axis = 0 )
                S.A = np.concatenate( (S.A, surf.A ) )
                S.ns = np.concatenate( (S.ns, surf.ns ) )
                S.reflectivity = np.concatenate( (S.reflectivity, surf.reflectivity) )
                S.eta = np.concatenate( (S.eta, surf.eta ) )
                S.FOV = np.concatenate( (S.FOV, surf.FOV ) )
                
        return S

# =============================================================================
# A rectangular room class containing 6 surfaces corresponding to the walls           
# =============================================================================            
class rectangular_room(room):
    
        def __init__(self, *args, length = None, width = None, height = None, D = None, **kwargs):                        
            names = ['ceiling', 'floor', 'north', 'south', 'east', 'west']
            corners = None
                
            for name in names:
                corner = room.corners( name, height = height, width = width, length = length ) 
                
                if corners is None:
                    corners = corner
                else:
                    corners = np.concatenate( (corners, corner), axis = 0)
                    
            self.length = length
            self.width = width
            self.height = height
            super().__init__(*args, corners = corners, names = names, D = D, **kwargs)       

# =============================================================================
# Impulse response class           
# =============================================================================
class impulse_response_simulation:
    
    def __init__(self, R = None, receiver_surface = None, time_axis = None, transmitter = None,
                 no_of_rays = 10000, no_of_bounces = 4, track_rays = False, show_output = True,
                 show_output_every = 100):

        self.R = R
        self.receiver_surface = receiver_surface
        self.transmitter = transmitter
        self.show_output_every = show_output_every
        
        if time_axis is None:
            time_axis = np.arange(0.0, 100.0e-9, 0.2e-9)
        
        self.time_axis = time_axis
        self.Dt = time_axis[1] - time_axis[0]
        self.no_of_rays = no_of_rays
        self.no_of_bounces = no_of_bounces
        self.track_rays = True
        self.show_output = show_output
                
        if receiver_surface is not None:
            
            self.bin = np.zeros( [receiver_surface.number_of_points(), time_axis.size] )
            
            
    def sim_print(self,*args,**kwargs):
        if self.show_output: 
            print(*args,**kwargs)
                
    def ray_trace(self):

        self.sim_print('Beginning ray tracing simulation')    
        self.tstart = time.time()
        
        self.sim_print('Generating ray orientations')
        self.normal = self.transmitter.normal[0,:]
        self.origin = self.transmitter.points[0,:]
        
        self.ray_orientations = generate_orientation_vectors( self.no_of_rays, self.normal )
        self.bin_bounces = np.zeros( [self.receiver_surface.number_of_points(), 
                                      self.time_axis.size,
                                      self.no_of_bounces] )
           
        point_range = range( self.receiver_surface.number_of_points() )
        self.dropped_rays = 0
        
        for ray_index in range(1,self.no_of_rays+1):
            
            if np.mod(ray_index, self.show_output_every) == 0 :
                self.sim_print('Generating ray %d out of %d' %(ray_index, self.no_of_rays) )

            
            r =  ray( position = self.origin,
                      orientation = self.ray_orientations[ray_index-1],
                      power = 1.0 / self.no_of_rays )   
            
            p = self.origin
            n = self.normal
            
            for bounce_index in range(1,self.no_of_bounces+1):
                
                st = single_point( point = r.position, normal = n, 
                                   power = r.power )
                
                power = self.receiver_surface.calc_power_from(st, update_surface = False)

                time_in_flight = ( r.time + self.receiver_surface.distance_from_point( p ) ) / const.c
                time_indexes = np.round(time_in_flight / self.Dt).astype(int)
                
                self.bin[ point_range, time_indexes ] += power
                self.bin_bounces[ point_range, time_indexes, bounce_index-1 ] += power
                _, p, n = r.bounce( self.R )
                
                if p is None:
                    self.dropped_rays += 1
                    break
    
    def power_at_surfaces(self):
        
        for key in self.R.surfaces:
            
            self.R.surfaces[key].calc_power_from(self.transmitter, update_surface = True)
            
    def barry_1st_order_response(self):
        
        all_elems = R.concatenate_surfaces()
        all_elems.set_power(1)
        t = np.zeros(all_elems.points.shape)
        
        # distances from transmitter and receiver
        dt = all_elems.distance_from_surface(self.transmitter)        
        dr = all_elems.distance_from_surface(self.receiver_surface)
        dt = dt.reshape( dt.size )
        dr = dr.reshape( dr.size )
        
        # DC components from transmitter and receiver
        Ht = all_elems.power_contributions_from(self.transmitter)
        Hr = self.receiver_surface.power_contributions_from(all_elems)
        Ht = Ht.reshape( Ht.size )
        Hr = Ht.reshape( Hr.size )
                
        # Transmit power from transmitter to elements
        p = Ht * all_elems.reflectivity * Hr
        d = (dt + dr)/const.c
        i = np.round(d/self.Dt).astype(int)
        S.bin[0,:] = arr_sum(p, i, self.time_axis.size)         
        print( np.max(d) )
        
    
                
print('Starting simulation...')
print('Setting room surfaces')
R = rectangular_room(length = 5.0, width = 5.0, height = 3.0, D = 0.1)
R.surfaces['north'].set_reflectivity(0.8)
R.surfaces['west'].set_reflectivity(0.8)
R.surfaces['east'].set_reflectivity(0.8)
R.surfaces['south'].set_reflectivity(0.8)
R.surfaces['ceiling'].set_reflectivity(0.8)
R.surfaces['floor'].set_reflectivity(0.3)
print('Room surfaces set')
#
transmitter = single_point(point = np.array([2.5, 2.5, 3.0]), 
                           normal = np.array([0.0, 0.0, -1.0]), power = 1.0)

receiver = single_point( point = np.array([ 0.5, 1.0, 0]),
                         normal = np.array([ 0.0, 0.0, 1.0]),
                         A = 1e-4,
                         FOV = 85/90 * np.pi/2.0) 

all_elems = R.concatenate_surfaces()
d = all_elems.distance_from_surface(transmitter)
pc = all_elems.power_contributions_from(transmitter)
t = d / const.c


print('Starting impulse response simulations')
S = impulse_response_simulation(R = R, receiver_surface = receiver, 
                                transmitter = transmitter, 
                                no_of_rays = 10000, 
                                no_of_bounces = 4)

S.ray_trace()
fig = plt.figure()
h = S.bin[0,:] / S.Dt
plt.plot(S.time_axis/1e-9, h)
plt.ylim([0.0, 300])
plt.ylabel('$h(t)$')
plt.xlabel('$t$ [ns]')
plt.savefig('ht.png')

