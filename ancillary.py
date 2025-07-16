# These are ancillary files for the analytical calculations in the Appendix
# of Hwang and Zakamska (2025)
# Written by Nadia Zakamska June 2024

import numpy as np
import sys
import mcint
import random

#************************************************************************
#************************************************************************
#************************************************************************

def calculate_normalization(prob,t_ms,mmin,mmax,fbinaries=0.05,present_day=12.0,ntrials=100000,tol=0.1):
    # calculate the normalization, its error, 
    # birth binary fraction and its error
    def integrand_bb(x):
        # x[0] is time and x[1] and x[2] are masses
        tprime=x[0]
        mass1=x[1]
        mass2=x[2]
        H1=0.5*(np.sign(t_ms(mass1)-(present_day-tprime))+1)
        H2=0.5*(np.sign(t_ms(mass2)-(present_day-tprime))+1)
        return prob(mass1)*prob(mass2)*H1*H2
    def integrand_bsbb(x):
        # x[0] is time and x[1] and x[2] are masses
        tprime=x[0]
        mass1=x[1]
        mass2=x[2]
        H1=0.5*(np.sign(t_ms(mass1)-(present_day-tprime))+1)
        H2=0.5*(np.sign(t_ms(mass2)-(present_day-tprime))+1)
        return prob(mass1)*prob(mass2)*H1
    def integrand_s(x):
        # x[0] is time and x[1] is mass
        tprime=x[0]
        mass=x[1]
        H=0.5*(np.sign(t_ms(mass)-(present_day-tprime))+1)
        return prob(mass)*H
    domainsize_b=present_day*(mmax-mmin)**2
    domainsize_s=present_day*(mmax-mmin)

    def sampler_b():
        while True:
            tt = random.uniform(0., present_day)
            mm1 = random.uniform(mmin, mmax)
            mm2 = random.uniform(mmin, mmax)
            yield (tt, mm1, mm2)
    def sampler_s():
        while True:
            tt = random.uniform(0., present_day)
            mm = random.uniform(mmin, mmax)
            yield (tt, mm)

    bb, error_bb = mcint.integrate(integrand_bb, sampler_b(), measure=domainsize_b, n=ntrials)
    if (error_bb/bb>tol): bb, error_bb = mcint.integrate(integrand_bb, sampler_b(), measure=domainsize_b, n=ntrials*10)
    bsbb, error_bsbb = mcint.integrate(integrand_bsbb, sampler_b(), measure=domainsize_b, n=ntrials)
    if (error_bsbb/bsbb>tol): bsbb, error_bsbb = mcint.integrate(integrand_bsbb, sampler_b(), measure=domainsize_b, n=ntrials*10)
    s, error_s = mcint.integrate(integrand_s, sampler_s(), measure=domainsize_s, n=ntrials)
    if (error_s/s>tol): s, error_s = mcint.integrate(integrand_s, sampler_s(), measure=domainsize_s, n=ntrials*10)
    print('fractional errors: s, bsbb, bb', error_s/s, error_bsbb/bsbb, error_bb/bb)
    AS=2*(bb-fbinaries*bsbb)/(fbinaries*s)
    error_AS=np.sqrt((2*error_bb/(fbinaries*s))**2+(2*error_bsbb/s)**2+(AS*error_s/s)**2)
    print('theoretical normalization, error, relative error', AS, error_AS, error_AS/AS)
    birth_fbinaries=2/(AS+2)
    error_birth_fbinaries=2*error_AS/(AS+2)**2
    return(AS,error_AS,birth_fbinaries,error_birth_fbinaries)

#************************************************************************
#************************************************************************
#************************************************************************

def calculate_fraction(prob,t_ms,mmin,mmax,mass1,AS,error_AS,present_day=12.0,ntrials=100000,tol=0.1):
    # calculate the theoretical binary fraction and its error for mass1
    def sub_bb(x):
        # x[0] is time and x[1] is mass
        tprime=x[0]
        mass2=x[1]
        H1=0.5*(np.sign(t_ms(mass1)-(present_day-tprime))+1)
        H2=0.5*(np.sign(t_ms(mass2)-(present_day-tprime))+1)
        return prob(mass2)*H1*H2
    def sub_bsbb(x):
        # x[0] is time and x[1] is mass
        tprime=x[0]
        mass2=x[1]
        H1=0.5*(np.sign(t_ms(mass1)-(present_day-tprime))+1)
        H2=0.5*(np.sign(t_ms(mass2)-(present_day-tprime))+1)
        return prob(mass2)*H1
    def sub_s(x):
        # x[0] is time
        tprime=x
        H1=0.5*(np.sign(t_ms(mass1)-(present_day-tprime))+1)
        return H1
    domainsize_b=present_day*(mmax-mmin)
    domainsize_s=present_day

    def sampler_b():
        while True:
            tt = random.uniform(0., present_day)
            mm2 = random.uniform(mmin, mmax)
            yield (tt, mm2)
    def sampler_s():
        while True:
            tt = random.uniform(0., present_day)
            yield (tt)

    bb, error_bb = mcint.integrate(sub_bb, sampler_b(), measure=domainsize_b, n=ntrials)
    if (error_bb/bb>tol): bb, error_bb = mcint.integrate(sub_bb, sampler_b(), measure=domainsize_b, n=ntrials*10)
    bsbb, error_bsbb = mcint.integrate(sub_bsbb, sampler_b(), measure=domainsize_b, n=ntrials)
    if (error_bsbb/bsbb>tol): bsbb, error_bsbb = mcint.integrate(sub_bsbb, sampler_b(), measure=domainsize_b, n=ntrials*10)
    s, error_s = mcint.integrate(sub_s, sampler_s(), measure=domainsize_s, n=ntrials)
    if (error_s/s>tol): s, error_s = mcint.integrate(sub_s, sampler_s(), measure=domainsize_s, n=ntrials*10)
    print('fractional errors', error_s/s, error_bsbb/bsbb, error_bb/bb)
    bin_fraction=2*bb/(AS*s+2*bsbb)
    error_bin_fraction=np.sqrt((bin_fraction*s*error_AS/(AS*s+2*bsbb))**2+(bin_fraction*error_s*AS/(AS*s+2*bsbb))**2+(bin_fraction*2*error_bsbb/(AS*s+2*bsbb))**2+(bin_fraction*error_bb/bb)**2)
    print('bin_fraction, error, rel. error', bin_fraction, error_bin_fraction, error_bin_fraction/bin_fraction)
    return(bin_fraction, error_bin_fraction)

#************************************************************************
#************************************************************************
#************************************************************************

def calculate_classes(prob,t_ms,mmin,mmax,present_day=12.0,ntrials=100000,tol=0.1):
    # given a mass distribution:
    # calculate the fraction of singles that are alive (class 0 vs class 2)
    # calculate the fraction of binaries which are dead, half-dead and alive 
    # (class 0, 1 and 2)
    def integrand_bb(x):
        # x[0] is time and x[1] and x[2] are masses
        tprime=x[0]
        mass1=x[1]
        mass2=x[2]
        H1=0.5*(np.sign(t_ms(mass1)-(present_day-tprime))+1)
        H2=0.5*(np.sign(t_ms(mass2)-(present_day-tprime))+1)
        return prob(mass1)*prob(mass2)*H1*H2
    def integrand_bs(x):
        # x[0] is time and x[1] and x[2] are masses
        tprime=x[0]
        mass1=x[1]
        mass2=x[2]
        H1=0.5*(np.sign(t_ms(mass1)-(present_day-tprime))+1)
        H2=0.5*(np.sign(t_ms(mass2)-(present_day-tprime))+1)
        return prob(mass1)*prob(mass2)*(H1+H2-2*H1*H2)
    def integrand_bd(x):
        # x[0] is time and x[1] and x[2] are masses
        tprime=x[0]
        mass1=x[1]
        mass2=x[2]
        H1=0.5*(np.sign(t_ms(mass1)-(present_day-tprime))+1)
        H2=0.5*(np.sign(t_ms(mass2)-(present_day-tprime))+1)
        return prob(mass1)*prob(mass2)*(1-H1)*(1-H2)
    def integrand_s(x):
        # x[0] is time and x[1] is mass
        tprime=x[0]
        mass=x[1]
        H=0.5*(np.sign(t_ms(mass)-(present_day-tprime))+1)
        return prob(mass)*H
    domainsize_b=present_day*(mmax-mmin)**2
    domainsize_s=present_day*(mmax-mmin)

    def sampler_b():
        while True:
            tt = random.uniform(0., present_day)
            mm1 = random.uniform(mmin, mmax)
            mm2 = random.uniform(mmin, mmax)
            yield (tt, mm1, mm2)
    def sampler_s():
        while True:
            tt = random.uniform(0., present_day)
            mm = random.uniform(mmin, mmax)
            yield (tt, mm)

    bb, error_bb = mcint.integrate(integrand_bb, sampler_b(), measure=domainsize_b, n=ntrials)
    if (error_bb/bb>tol): bb, error_bb = mcint.integrate(integrand_bb, sampler_b(), measure=domainsize_b, n=ntrials*10)
    bs, error_bs = mcint.integrate(integrand_bs, sampler_b(), measure=domainsize_b, n=ntrials)
    if (error_bs/bs>tol): bs, error_bs = mcint.integrate(integrand_bs, sampler_b(), measure=domainsize_b, n=ntrials*10)
    bd, error_bd = mcint.integrate(integrand_bd, sampler_b(), measure=domainsize_b, n=ntrials)
    if (error_bd/bd>tol): bd, error_bd = mcint.integrate(integrand_bd, sampler_b(), measure=domainsize_b, n=ntrials*10)
    s, error_s = mcint.integrate(integrand_s, sampler_s(), measure=domainsize_s, n=ntrials)
    if (error_s/s>tol): s, error_s = mcint.integrate(integrand_s, sampler_s(), measure=domainsize_s, n=ntrials*10)
    print('fractional errors: bb, bs, bd, s', error_bb/bb, error_bs/bs, error_bd/bd, error_s/s)
    return(bb/present_day, bs/present_day, bd/present_day, s/present_day, error_bb/present_day, error_bs/present_day, error_bd/present_day, error_s/present_day)

