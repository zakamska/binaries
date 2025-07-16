# These are functions that are used for orbital dynamics calculations 
# in Hwang and Zakamska (2025)
# Written by Nadia Zakamska June 2024

# to reload in python 3.6:
# import importlib
# importlib.reload(orbital)

# list of functions:
#def per(mass,a):
#def sma(period,m1,m2): given period in days and masses, return sma in AU
#def scal(a,b):
#def testing_scal():
#def mylen(x):
#def true_from_ecc_old(ecc,ecc_anomaly):
#def true_from_ecc(ecc,ecc_anomaly):
#def ecc_from_true(ecc,true_anomaly):
#def mean_from_ecc(ecc,ecc_anomaly):
#def ecc_from_mean(ecc,mean_anomaly):
#def testing_anomalies():
#def orbital_from_cart(m,r,v):
#def testing_orbital_from_cart():
#def cart_from_orbital(m, a, ecc, inc, asclong, omega, mean_anomaly):
#def testing_cart_from_orbital():
#def one_kick_loss(m1,m2,deltam2,vkick,orbit):
    # it's m2 that's losing mass in the amount deltam and it's the one getting kicked
#def testing_one_kick_loss():

import numpy as np
import scipy
import scipy.optimize
from astropy.table import Table

# **********************************************************************
# **********************************************************************
# **********************************************************************

# the range of the orbital angles:
# inclination is from 0 to pi (Murray and Dermott Section 2.8)
# if inclination is <pi/2 then prograde, if inclination is >pi/2 then 
# retrograde
# omega, asclong, mean_anomaly are from 0 to 2pi

# **********************************************************************
# **********************************************************************
# **********************************************************************

# Some essentials

kms=1e5
year=365.*24.*3600.
G=6.67e-8
Msun=2e33
AU=1.5e13
deg=np.pi/180.
day=24.*3600.

def per(mass,a):
    # total mass in Msun, a in AU
    # returns period in years
    return(2*np.pi*np.sqrt((a*AU)**3/(G*Msun*mass))/year)

def sma(period,m1,m2):
    # given period in days and m1 and m2 in solar masses, compute semi-major axis in AU
    koef=(day/(2*np.pi))**(2/3)*(G*Msun)**(1/3)/AU
    return(koef*period**(2/3)*(m1+m2)**(1/3))

def scal(a,b):
    # https://stackoverflow.com/questions/37670658/python-dot-product-of-each-vector-in-two-lists-of-vectors
    # return a list of pairwise scalar products of vectors from two lists
    # np.shape(a)=np.shape(b)=(length_of_list, 3)
    # unfortunately, for just two vectors, they must be reshaped: scal(np.array([[1.,2.,3.]]),np.array([[2.,3.,4]]))
    return(np.einsum('ij, ij->i', a, b))

def testing_scal():
    a=np.ones((2,3))
    a[1,2]=5
    b=np.ones((2,3))*2
    b[0,0]=0
    print('a: ', a)
    print('b: ', b)
    print('vector product: ', np.cross(a,b))
    print('scalar product: ', scal(a,b))
    print('projection onto z axis:', a.dot(np.transpose(np.array([0,0,1]))))
    return(1)
    #output:
    #a:  [[1. 1. 1.]
    #     [1. 1. 5.]]
    #b:  [[0. 2. 2.]
    #     [2. 2. 2.]]
    #vector product:  [[ 0. -2.  2.]
    #                  [-8.  8.  0.]]
    #scalar product:  [ 4. 14.]
    #projection onto z axis: [1. 5.]

def mylen(x):
    # return 1 if a scalar and length otherwise
    if len(np.shape(x))==0: return 1
    return len(x)

# **********************************************************************
# **********************************************************************
# **********************************************************************

# Conversions of various anomalies

def true_from_ecc_old(ecc,ecc_anomaly):
    # given eccentricity and ecc_anomaly, return true anomaly
    # works for scalars and for vectors
    ecc=np.array([ecc])
    ecc_anomaly=np.array([ecc_anomaly])
    true_anomaly=np.arccos((np.cos(ecc_anomaly)-ecc)/(1.-ecc*np.cos(ecc_anomaly)))
    ind=(ecc_anomaly>np.pi)
    true_anomaly[ind]=2*np.pi-true_anomaly[ind]
    return(true_anomaly[0])

def true_from_ecc(ecc,ecc_anomaly):
    # given eccentricity and ecc_anomaly, return true anomaly
    # works for scalars and for vectors
    # if eccentricitiy is >1, return the value of true_anomaly
    ecc=np.array([ecc])
    # create temporary eccentricity so we don't get errors for ecc>1
    tecc=ecc
    tecc[(np.abs(ecc)>1.0)]=0.
    ecc_anomaly=np.array([ecc_anomaly])
    true_anomaly=np.arctan2(np.sqrt(1-tecc**2)*np.sin(ecc_anomaly)/(1-tecc*np.cos(ecc_anomaly)),(np.cos(ecc_anomaly)-tecc)/(1-tecc*np.cos(ecc_anomaly)))
    ind=(true_anomaly<0)
    true_anomaly[ind]=2*np.pi+true_anomaly[ind]
    return(true_anomaly[0])

# testing commands:
#orbital.true_from_ecc(0.2,0.1)
#0.12242351980596208
#orbital.true_from_ecc([0.,0.1],0.1)
#array([0.1      , 0.1105337])
#orbital.true_from_ecc(0.1,[0.,0.1])
#array([0.       , 0.1105337])
#orbital.true_from_ecc([0.,0.2],[0.,0.1])
#array([0.        , 0.12242352])
#orbital.true_from_ecc([1.5, 0.2], [0., 0.1])
# array([0.        , 0.12242352])

def ecc_from_true(ecc,true_anomaly):
    # given eccentricity and true anomaly, return eccentric anomaly
    # works for scalars and for vectors
    # if eccentricitiy is >1, return the value of true_anomaly
    ecc=np.array([ecc])
    # create temporary eccentricity so we don't get errors for ecc>1
    tecc=ecc
    tecc[(np.abs(ecc)>1.0)]=0.
    true_anomaly=np.array([true_anomaly])
    ecc_anomaly=np.arctan2(np.sqrt(1-tecc**2)*np.sin(true_anomaly)/(1+tecc*np.cos(true_anomaly)), (np.cos(true_anomaly)+tecc)/(1+tecc*np.cos(true_anomaly)))
    # arctan2 function is defined on -pi to pi
    ind=(ecc_anomaly<0)
    ecc_anomaly[ind]=ecc_anomaly[ind]+2*np.pi
    return(ecc_anomaly[0])

#orbital.ecc_from_true(0.2,0.1224235198)
#0.0999999999951259
#orbital.ecc_from_true([0.,0.1],0.1105337)
#array([0.1105337, 0.1      ])
#orbital.ecc_from_true(0.1, [0.       , 0.1105337])
#array([0. , 0.1])
#orbital.ecc_from_true([0.,0.2],[0.        , 0.12242352])
#array([0. , 0.1])

def mean_from_ecc(ecc,ecc_anomaly):
    # given eccentricy and eccentric anomaly, return mean anomaly
    # works for arrays, too
    return(ecc_anomaly-ecc*np.sin(ecc_anomaly))

def ecc_from_mean(ecc,mean_anomaly):
    # given eccentricy and mean anomaly, return eccentric anomaly
    # works for arrays, too
    # if ecc>1, return mean_anomaly
    if (mylen(ecc)>1 and mylen(mean_anomaly)==1): mean_anomaly=np.ones(len(ecc))*mean_anomaly
    tecc=ecc
    if (mylen(ecc)==1 and ecc>1.): tecc=0.
    if (mylen(ecc)>1): tecc[(ecc>1.)]=0.
    def myfunc(x):
        return(x-tecc*np.sin(x)-mean_anomaly)
    ecc_anomaly = scipy.optimize.newton_krylov(myfunc, mean_anomaly, f_tol=1e-14)
    return(ecc_anomaly)

# testing commands:
# orbital.ecc_from_mean(0.2,0.1)
# array(0.12491884)
# orbital.ecc_from_mean([0.,0.2],[0.1,0.1])
# array([0.1       , 0.12491884])
# orbital.ecc_from_mean(0.2,[0.,0.1])
# array([0.        , 0.12491884])
# orbital.ecc_from_mean([0.,0.2],0.1)
# array([0.1       , 0.12491884])

def test_anomalies():
    mean_anomaly=np.arange(0,4*np.pi,0.01)
    ecc=0.5
    ecc_anomaly=orbital.ecc_from_mean(ecc,mean_anomaly)
    true_anomaly=orbital.true_from_ecc(ecc,ecc_anomaly)
    fig=plt.figure()
    plt.plot(mean_anomaly,ecc_anomaly, color='blue',label='eccentric anomaly')
    plt.plot(mean_anomaly,true_anomaly, color='red',label='true anomaly')
    plt.legend(loc='upper left')
    plt.xlabel('mean anomaly')
    plt.ylabel('other anomalies')
    fig.tight_layout()
    plt.show()
    ecc=1.5
    ecc_anomaly=orbital.ecc_from_mean(ecc,mean_anomaly)
    true_anomaly=orbital.true_from_ecc(ecc,ecc_anomaly)
    fig=plt.figure()
    plt.plot(mean_anomaly,ecc_anomaly, color='blue',label='eccentric anomaly')
    plt.plot(mean_anomaly,true_anomaly, color='red',label='true anomaly')
    plt.legend(loc='upper left')
    plt.xlabel('mean anomaly')
    plt.ylabel('other anomalies')
    fig.tight_layout()
    plt.show()
    return(1)

# **********************************************************************
# **********************************************************************
# **********************************************************************

def orbital_from_cart(m,r,v):
    # mass can be a scalar or a vector
    # r can be a single vector or a list of vectors, same with v
    # everything is in Rebound units here
    if (np.shape(r)==(3,)):
        r=np.array([r])
        v=np.array([v])
    # Euler transformations are here: https://en.wikipedia.org/wiki/Orbital_elements
    # instantaneous vector of angular momentum
    inst_J=np.cross(r,v)
    # instantaneous inclination if the angle between the angular momentum vector and the z axis, 
    mag_J=np.sqrt(scal(inst_J,inst_J))
    inst_inc=np.arccos(inst_J.dot(np.transpose(np.array([0,0,1])))/mag_J)
    norm_J=np.transpose(np.array([inst_J[:,0]/mag_J,inst_J[:,1]/mag_J,inst_J[:,2]/mag_J]))
    # the ratio of the first two components of this is -tan of asclong, and sin inc is always positive
    inst_asclong=np.arctan2(norm_J[:,0],-norm_J[:,1])
    # this is defined on -pi to pi, so fix:
    ind=(inst_asclong<0)
    inst_asclong[ind]=2*np.pi+inst_asclong[ind]
    # the semi-major axis comes from the energy: 
    mag_v=np.sqrt(scal(v,v))
    mag_r=np.sqrt(scal(r,r))
    kin_energy=0.5*mag_v**2
    pot_energy=-m/mag_r
    inst_en=kin_energy+pot_energy
    inst_sma=-m/(2*inst_en)
    # eccentricity comes from the combination of L and sma:
    inst_ecc=np.sqrt(1.-mag_J**2/(m*inst_sma))
    # Runge-Lenz vector points to the pericenter:
    inst_R=np.cross(v,inst_J)-np.transpose([m])*np.transpose(np.array([r[:,0]/mag_r,r[:,1]/mag_r,r[:,2]/mag_r]))
    mag_R=np.sqrt(scal(inst_R,inst_R))
    # unit vector directed to the pericenter
    norm_R=np.transpose(np.array([inst_R[:,0]/mag_R,inst_R[:,1]/mag_R,inst_R[:,2]/mag_R]))
    x1=norm_R[:,0]
    x2=norm_R[:,1]
    x3=norm_R[:,2] # x3=sin(inc)*sin(omega) from Euler transformations
    inst_omega=np.arctan2(x3/np.sin(inst_inc),x1*np.cos(inst_asclong)+x2*np.sin(inst_asclong))
    # this is defined on -pi to pi, so fix:
    ind=(inst_omega<0)
    inst_omega[ind]=2*np.pi+inst_omega[ind]
# normally here we would have inst_omega=np.arcsin(x3/np.sin(inst_inc))
# but apparently it is sometimes >1 or <-1 (presumably by a small amount)
#    temp=x3/np.sin(inst_inc)
#    ind=(temp>1.0)
#    temp[ind]=1.0
#    ind=(temp<-1.0)
#    temp[ind]=-1.0
#    inst_omega=np.arcsin(temp)
    # true anomaly: 
    cos_true=scal(r,inst_R)/(mag_r*mag_R)
    ind=(cos_true>1.)
    cos_true[ind]=1.0
    ind=(cos_true<-1.)
    cos_true[ind]=-1.0
    inst_true=np.arccos(cos_true)
    # check to see if the particle is approaching the pericenter or receding:
    ind=(scal(r,v)<0)
    inst_true[ind]=2*np.pi-inst_true[ind]
    # I can probably get one of the other anomalies from the rv angle
    # in case this one gives me any trouble!
    return(inst_sma,inst_ecc,inst_inc,inst_asclong,inst_omega,inst_true)

def testing_orbital_from_cart():
    rvec0=[-2361.44523602, -1272.65342042,   171.09549598]
    vvec0=[ 0.02459268, -0.04256368,  0.02282615]
    print(orbital_from_cart(1.0, np.array(rvec0), np.array(vvec0)))
    rvec=np.array([rvec0, rvec0])
    vvec=np.array([vvec0, vvec0])
    print(orbital_from_cart(np.array([1.,2.]),rvec,vvec))
    print(orbital_from_cart(1.0,rvec,vvec))
    return(1)

# *********************************************************************
# *********************************************************************
# *********************************************************************

def cart_from_orbital(m, a, ecc, inc, asclong, omega, mean_anomaly):
    # calculate Cartesian coordinates from orbital elements
    # should work for both scalars and vectors
    # DOES NOT WORK FOR ECC>1 BECAUSE IT CAN'T COMPUTE ECC_ANOMALY
    # if it's scalars, we need to convert to vectors
    # now the one concession we'll make is if mass is a scalar, it should
    # stay a scalar
    if (len(np.shape(a))==0): a=np.array([a])
    if (len(np.shape(ecc))==0): ecc=np.array([ecc])
    if (len(np.shape(inc))==0): inc=np.array([inc])
    if (len(np.shape(asclong))==0): asclong=np.array([asclong])
    if (len(np.shape(omega))==0): omega=np.array([omega])
    if (len(np.shape(mean_anomaly))==0): mean_anomaly=np.array([mean_anomaly])
    # compute orbit orientation relative to the fixed frame
    # https://en.wikipedia.org/wiki/Orbital_elements#Euler_angle_transformations
    x1=np.cos(asclong)*np.cos(omega)-np.sin(asclong)*np.cos(inc)*np.sin(omega)
    x2=np.sin(asclong)*np.cos(omega)+np.cos(asclong)*np.cos(inc)*np.sin(omega)
    x3=np.sin(inc)*np.sin(omega)
    e_x=np.transpose(np.array([x1,x2,x3]))
    y1=-np.cos(asclong)*np.sin(omega)-np.sin(asclong)*np.cos(inc)*np.cos(omega)
    y2=-np.sin(asclong)*np.sin(omega)+np.cos(asclong)*np.cos(inc)*np.cos(omega)
    y3=np.sin(inc)*np.cos(omega)
    e_y=np.transpose(np.array([y1,y2,y3]))
    z1=np.sin(inc)*np.sin(asclong)
    z2=-np.sin(inc)*np.cos(asclong)
    z3=np.cos(inc)
    e_z=np.transpose(np.array([z1,z2,z3]))
    # compute values in the orbital plane
    ecc_anomaly=ecc_from_mean(ecc,mean_anomaly)
    true_anomaly=true_from_ecc(ecc,ecc_anomaly)
    r=a*(1-ecc*np.cos(ecc_anomaly))
    # Compute the velocity vector and the separation vector in the fixed 3D frame
    rvec=np.transpose([r*np.cos(true_anomaly)])*e_x+np.transpose([r*np.sin(true_anomaly)])*e_y
    e_r=np.transpose(np.array([rvec[:,0]/r,rvec[:,1]/r,rvec[:,2]/r]))
    vphi=np.sqrt(m/a)*np.sqrt(1.-ecc**2)/(1.-ecc*np.cos(ecc_anomaly))
    vr=np.sqrt(m/a)*ecc*np.sin(ecc_anomaly)/(1.-ecc*np.cos(ecc_anomaly))
    vvec=np.transpose([vr*np.cos(true_anomaly)-vphi*np.sin(true_anomaly)])*e_x+np.transpose([vr*np.sin(true_anomaly)+vphi*np.cos(true_anomaly)])*e_y
    # confirming that these values agree with Rebound
    return(rvec, vvec)

def testing_cart_from_orbital():
    # function to make sure that everything works well 
    import rebound
    m=0.1
    a=1.
    ecc=0.1
    inc=0.1
    asclong=0.1
    omega=0.1
    mean_anomaly=0.1
    (rvec0, vvec0)=cart_from_orbital(m, a, ecc, inc, asclong, omega, mean_anomaly)
    print(rvec0,vvec0)
    sim = rebound.Simulation()
    sim.add(m=m)                  # Stationary center of mass
    sim.add(m=0., a=a, e=ecc, inc=inc, Omega=asclong, omega=omega, M=mean_anomaly) # moving object
    ps = sim.particles
    print(ps[1])
    m=np.array([1.0,5.0])
    a=np.array([1000.,2000.])
    ecc=np.array([0.1,0.3])
    inc=np.array([0.3,0.4])
    asclong=np.array([1.,2.])
    omega=np.array([0.,0.5])
    mean_anomaly=np.array([0.,0.4])
    (rvec0, vvec0)=cart_from_orbital(m, a, ecc, inc, asclong, omega, mean_anomaly)
    print(rvec0[0],vvec0[0])
    print(rvec0[1],vvec0[1])
    sim = rebound.Simulation()
    sim.add(m=m[0])                  # Stationary center of mass
    sim.add(m=0., a=a[0], e=ecc[0], inc=inc[0], Omega=asclong[0], omega=omega[0], M=mean_anomaly[0]) # moving object
    ps = sim.particles
    print(ps[1])
    sim = rebound.Simulation()
    sim.add(m=m[1])                  # Stationary center of mass
    sim.add(m=0., a=a[1], e=ecc[1], inc=inc[1], Omega=asclong[1], omega=omega[1], M=mean_anomaly[1]) # moving object
    ps = sim.particles
    print(ps[1])
    return(1)

#********************************************************************************
#********************************************************************************
#********************************************************************************

def one_kick_loss(m1,m2,deltam2,vkick,orbit):
    # it's m2 that's losing mass in the amount deltam
    # vkick is a 3D velocity vector that is imparted to m2, and it's in Rebound (code) units G=Msun=1
    # orbit it a dictionary (or a Table) with six columns
    # a, ecc, inc, asclong, omega, mean_anomaly
    # these are the initial vectors of the system: 
    # THIS RIGHT HERE WILL NOT WORK FOR ANY OBJECTS STARTING WITH ECC>1
    rvec,vvec=cart_from_orbital(m1+m2, orbit['sma'], orbit['ecc'], orbit['inc'], orbit['asclong'], orbit['omega'], orbit['mean_anomaly'])
    # these will be the final parameter of the system
    new_orbit=Table()
    # rvec doesn't change. vvec decreases by vkick if it's mass m2. 
    inst_sma,inst_ecc,inst_inc,inst_asclong,inst_omega,inst_true=orbital_from_cart(m1+m2-deltam2, rvec, vvec-vkick)
    new_orbit['sma']=inst_sma
    new_orbit['ecc']=inst_ecc
    new_orbit['inc']=inst_inc
    new_orbit['asclong']=inst_asclong
    new_orbit['omega']=inst_omega
    new_orbit['mean_anomaly']=mean_from_ecc(inst_ecc,ecc_from_true(inst_ecc,inst_true))
    return(new_orbit)

def testing_one_kick_loss():
    # testing the function with a direct integration
    from scipy.integrate import solve_ivp
    ain=8400. 
    eccin=0.68
    incin=0.44
    asclongin=3.5
    omegain=0.15
    m1=4.3
    m2=0.4
    total_mass=m1+m2
    # mass loss of the more massive star
    deltam=3.4
    m=m1+m2
    mean_an_in=0.
    orbit={'sma':np.array(ain), 'ecc':np.array(eccin), 'inc':np.array(incin), 'asclong':np.array(asclongin), 'omega':np.array(omegain), 'mean_anomaly':np.array(mean_an_in)}
    print(orbit, m1, m2)
    rvec0,vvec0=cart_from_orbital(m1+m2, orbit['sma'], orbit['ecc'], orbit['inc'], orbit['asclong'], orbit['omega'], orbit['mean_anomaly'])
    rvec0=rvec0[0]
    vvec0=vvec0[0]
    deltam=1.0
    vkick=0.01
    print(one_kick_loss(m2,m1,deltam,vkick*np.array([0.,0.,1.]),orbit))
    #now verify this with direct integration:
    # a hundred steps over 100 years... 
    nsteps=100
    deltat=100.0*tunit
    t_eval=np.linspace(0, deltat, nsteps+1)
    def tm(t):
        return(m1+m2-deltam*t/deltat)
    def ak(t):
        return(-vkick/deltat)
    def rhs(t, u): 
        r=np.sqrt(u[0]**2+u[1]**2+u[2]**2)
        return ([u[3],u[4],u[5],-tm(t)*u[0]/r**3,-tm(t)*u[1]/r**3,-tm(t)*u[2]/r**3+ak(t)])
    dir_res = solve_ivp(rhs, (0, deltat), [rvec0[0], rvec0[1], rvec0[2], vvec0[0], vvec0[1], vvec0[2]], t_eval=t_eval)
    print(orbital_from_cart(m1+m2-deltam,dir_res.y[0:3,nsteps],dir_res.y[3:6,nsteps]))
    return(1)

#******************************************************************
#******************************************************************
#******************************************************************

#https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
def max_likelihood_ecc(ecc, emax=1.0):
    # given an array of eccentricities, return the max likelihood alpha parameter
    # for the best-fit powerlaw distribution
    # if the eccentricities are only defined on 0 to emax, should also work
    tecc=ecc[(ecc<emax)]
    return(len(tecc)/(len(tecc)*np.log(emax)-np.sum(np.log(tecc)))-1.)

