import sys
import os
import numbers
import inspect
import math

import numpy as np
from numpy import linalg as la
import scipy
import scipy.integrate
import scipy.fft
import scipy.constants as constants
import scipy.special as special

debug = False

def _sqcoefOriginalHP(ir, eta, gek, ak, a=0., b=0., c=0., f=0., u=0., v=0., gamk=0., seta=0., sgek=0., sak=0., scal=0.,
                      g1=0.):
    """
    CALCULATES RESCALED VOLUME FRACTION AND CORRESPONDING COEFFICIENTS
    This is only for documenting the difference to the old algorithm.

    This is the iterative part to find rescaling parameter to get G(1+)>0 (Gillian condition) if G(1+)>0

    Returns:
    ir,eta,gek,ak,a,b,c,f,u,v,gamk,seta,sgek,sak,scal,g1

    seta IS THE RESCALED VOLUME FRACTION.
    sgek IS THE RESCALED CONTACT POTENTIAL.
    sak IS THE RESCALED SCREENING CONSTANT.
    a,b,c,f,u,v ARE THE MSA COEFFICIENTS.
    g1=G(1+) IS THE CONTACT VALUE OF G(R/SIG);
    FOR THE GILLAN CONDITION, THE DIFFERENCE FROM
    ZERO INDICATES THE COMPUTATIONAL ACCURACY.

    IR > 0: NORMAL EXIT, IR IS THE NUMBER OF ITERATIONS.
    < 0: FAILED TO CONVERGE.

    This is equivalent to the original HP Fortran code.
    The different conditions might have saved computing time in 1981.
    For some parameter conditions the rescaling is needed but not done.

    Also for some parameter contributions the wrong root for Fwww is used.

    """
    # set to zero to get debug messages; debuglevel>10 no messages
    debuglevel = 1
    itm = 40  # original 40
    acc = 5.e-6
    if debug > debuglevel: print('-- ')
    if ak >= (1 + 8. * eta):
        # for large screening (scl is small and ak is large)
        # ix=1  SOLVE FOR LARGE K, RETURN G(1+)
        ix, ir, g1, eta, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1 = \
            _sqfun(1, ir, g1, eta, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1, useHP=True)
        if debug > debuglevel: print('large screening ', ir, g1, ak, gamk, 'abcfuv', a, b, c, f, u, v)
        if ir < 0 or g1 >= 0:  # error or already a good solution is returned
            return ir, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1
        else:
            # we have to rescale the solution in the later as here g+<0
            pass
    seta = min(eta, 0.2)
    if ak >= (1 + 8. * eta) or gamk >= 0.15:
        # find a rescaled eta with g+>=0 for strong coupling or low volume fraction
        j = 0.
        f1 = 0.
        f2 = 0.
        while True:  # loop for Newton iteration to find g+=0
            j += 1
            if j > itm:
                return -1, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1
            if seta <= 0.0: seta = eta / j  # g+<0 -> rescale eta
            if seta > 0.6: seta = 0.35 / j  # rescaled eta>0.6 rescale to smaller value
            e1 = seta  # e1 first eta
            # ix=2  RETURN FUNCTION TO SOLVE FOR ETA(GILLAN)
            ix, ir, f1, e1, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1 = \
                _sqfun(2, ir, f1, e1, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1, useHP=True)
            e2 = seta * 1.01  # increase scaled eta
            ix, ir, f2, e2, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1 = \
                _sqfun(2, ir, f2, e2, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1, useHP=True)
            e2 = e1 - (e2 - e1) * f1 / (f2 - f1)  # new approximation for scaled eta
            seta = e2  # save for next iteration or as result
            delta = abs((e2 - e1) / e1)  # relative change
            if delta < acc: break  # if changes are small enough then break
        if debug > debuglevel: print('rescaling with %i iterations leads to scaling by %.3g' % (j, seta / eta))
        # ix=4    RETURN G(1+) FOR ETA=ETA(GILLAN).
        ix, ir, g1, e2, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g11 = \
            _sqfun(4, ir, g1, e2, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1, useHP=True)
        ir = j
        # ---------------end of Newton loop
        if debug > debuglevel: print('rescaled ', ir, g1, ak, gamk, 'abcfuv', a, b, c, f, u, v, 'ak>,seta>eta ',
                                     ak >= (1 + 8. * eta), seta >= eta)
        if ak >= (1 + 8. * eta):  # in this case return anyway
            return ir, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1
        else:
            if seta >= eta:  # seta>eta indicates successful rescaling with g1 as zero
                return ir, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1

    ix, ir, g1, eta, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1 = \
        _sqfun(3, ir, g1, eta, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1, useHP=True)
    if debug > debuglevel: print('after scaling ', ir, g1, ak, gamk, 'abcfuv', a, b, c, f, u, v)
    if ir >= 0:
        if g1 < 0.: ir = -3  # rescaling not successful
    return ir, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1


def _sqcoef(ir, eta, gek, ak, a=0., b=0., c=0., f=0., u=0., v=0., gamk=0., seta=0., sgek=0., sak=0., scal=0., g1=0.):
    """
    CALCULATES RESCALED VOLUME FRACTION AND CORRESPONDING COEFFICIENTS

    This is the iterative part to find rescaling parameter to get G(1+)>0 (Gillian condition) if G(1+)>0

    Returns:
    ir,eta,gek,ak,a,b,c,f,u,v,gamk,seta,sgek,sak,scal,g1

    seta IS THE RESCALED VOLUME FRACTION.
    sgek IS THE RESCALED CONTACT POTENTIAL.
    sak IS THE RESCALED SCREENING CONSTANT.
    a,b,c,f,u,v ARE THE MSA COEFFICIENTS.
    g1=G(1+) IS THE CONTACT VALUE OF G(R/SIG);
    FOR THE GILLAN CONDITION, THE DIFFERENCE FROM
    ZERO INDICATES THE COMPUTATIONAL ACCURACY.

    IR > 0: NORMAL EXIT, IR IS THE NUMBER OF ITERATIONS.
    < 0: FAILED TO CONVERGE.

    This is a shorter version of sqcoef which is easier to understand and allows
    no bypassing between the conditions in original code which leads to errors for harmless parameter settings.
    The idea is the original idea (see [2]_) to calculate the MSA and to rescale if  g+<0  .


    """
    # set to zero to get debug messages; debuglevel>10 no messages
    debuglevel = 1
    itm = 80  # original 40
    acc = 5.e-6
    fix = 0.5
    if debug > debuglevel: print('-- ')
    # just try to solve
    ix, ir, g1, eta, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1 = \
        _sqfun(1, ir, g1, eta, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1)
    if debug > debuglevel: print('first try ', ir, g1, ak, gamk, 'abcfuv', a, b, c, f, u, v)
    if ir == -2:
        # FAILED TO CONVERGE in Newton algorith to find zero, only in classical HP solution,
        return ir, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1
    elif ir == -4:
        # no root found in first try
        return ir, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1
    elif g1 < 0:
        # we have to rescale the solution in the later as here g+<0
        pass
    elif g1 >= 0:  # already a good solution is returned
        return ir, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1

    seta = min(eta, 0.2)
    # find a rescaled eta with g+>=0 for strong coupling or low volume fraction
    j = 0.
    f1 = 0.
    f2 = 0.
    while True:  # loop for Newton iteration to find g+=0
        j += 1
        if j > itm:
            return -1, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1
        if seta <= 0.0: seta = eta / j  # g+<0 -> rescale eta
        if seta > 0.6: seta = 0.35 / j  # rescaled eta>0.6 rescale to smaller value
        e1 = seta  # e1 first eta
        # ix=2  RETURN FUNCTION TO SOLVE FOR ETA(GILLAN)
        ix, ir, f1, e1, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1 = \
            _sqfun(2, ir, f1, e1, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1)
        e2 = seta * 1.01  # increase scaled eta
        ix, ir, f2, e2, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1 = \
            _sqfun(2, ir, f2, e2, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1)
        e2 = e1 - (e2 - e1) * f1 / (f2 - f1)  # new approximation for scaled eta
        seta = e2  # save for next iteration or as result
        delta = abs((e2 - e1) / e1)  # relative change
        if delta < acc: break  # changes  are small enough then break
    if debug > debuglevel: print('rescaling with %i iterations leads to scaling by %.3g' % (j, seta / eta))
    # ix=4    RETURN G(1+) FOR ETA=ETA(GILLAN) with all parameters.
    ix, ir, g1, e2, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g11 = \
        _sqfun(4, ir, g1, e2, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1)
    if (seta > 0.64) or (seta < eta):
        ir = -3  # rescaling not successful
        return ir, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1
    ir = j
    # ---------------end of Newton loop
    if debug > debuglevel: print('rescaled ', ir, g1, ak, gamk, 'abcfuv', a, b, c, f, u, v, 'ak>,seta,eta ',
                                 ak >= (fix + 8. * eta), seta, eta)
    return ir, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1


def _sqfun(ix, ir, fval, evar, reta, rgek, rak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1, useHP=False):
    """
    CALCULATES VARIOUS COEFFICIENTS AND FUNCTION VALUES FOR _sqcoef

    this is the NOT rescaled solution! == MSA

    Options
    ix =1: SOLVE FOR LARGE K, RETURN G(1+).
        2: RETURN FUNCTION TO SOLVE FOR ETA(GILLAN).
        3: ASSUME NEAR GILLAN, SOLVE, RETURN G(1+).
        4: RETURN G(1+) FOR ETA=ETA(GILLAN).

    SETA IS THE RESCALED VOLUME FRACTION.
    SGEK IS THE RESCALED CONTACT POTENTIAL.
    SAK IS THE RESCALED SCREENING CONSTANT.
    A,B,C,F,U,V ARE THE MSA COEFFICIENTS.
    G1=G(1+) IS THE CONTACT VALUE OF G(R/SIG);
    FOR THE GILLAN CONDITION, THE DIFFERENCE FROM
    ZERO INDICATES THE COMPUTATIONAL ACCURACY.

    IR > 0: NORMAL EXIT, IR IS THE NUMBER OF ITERATIONS.
     < 0: FAILED TO CONVERGE.

    The root of the quartic F = w4*fa**4+w3*fa**3+w2*fa**2+w1*fa+w0 needs to be found.
    in this code we have two choices in the source code.
    One for documentation and the second as the correct solution:

     1. to use the original HayterPenfold algorithm from the Fortran code as also eg used in SASVIEW and SASFIT
        with an estimate for the root of Fwww which is refined by Newton algorithm
        which results under specific conditions in the wrong root
        test with e.g.
        for scl in np.r_[1:10]:p.plot(js.sf.RMSA(q=x,R=3.1,scl=scl, gamma=1.1, eta=0.5),legend='%.3g' %scl)
        the correct branch can be verified by using the Percus-Yevick as limit

     2. original idea from Hayter paper [1]_ as *default  solution*
        find all roots (by numpy.roots) and take the physical root with g(r/diameter<1)=0
        in this code there is no difference between ix=1 and 3
        with structurefactor.debug=11 you get output for g(r) and the zeros of Fwww (see source code)



    """
    # set to zero to get debug messages; debuglevel>10 no messages
    debuglevel = 1
    acc = 1e-6  # stop criterion for Newton
    itm = 40  # max number of iterations
    # needed parameters with changes for iteration
    eta = evar  # volume fraction
    scal = (reta / evar) ** (1 / 3.)  # scaling factor
    sak = rak / scal  # scaled dimensionless screening constant
    val = rgek if abs(rgek) > 1e-9 else 1e-9  # prevent zero and just take small value
    sgek = val * scal * math.exp(rak - sak)  # scaled contact potential
    gek = sgek
    ak = sak
    # -----------------reproduce original fortran code
    # using these variables is important to reduce the dependency on accuracy of float64
    # and maybe it makes it a bit faster
    eta2 = eta ** 2
    eta3 = eta2 * eta
    e12 = 12. * eta
    e24 = e12 + e12
    ak2 = ak ** 2
    ak1 = 1 + ak
    dak2 = 1.0 / ak2
    dak4 = dak2 * dak2
    d = 1 - eta
    d2 = d * d
    dak = d / ak
    dd2 = 1.0 / d2
    dd4 = dd2 * dd2
    dd45 = dd4 * 2.0e-1
    eta3d = 3. * eta
    eta6d = eta3d + eta3d
    eta32 = eta3 + eta3
    eta2d = eta + 2.0
    eta2d2 = eta2d * eta2d
    eta21 = 2.0 * eta + 1.0
    eta22 = eta21 * eta21

    # all coefficients from appendix in the paper [1]
    al1 = -eta21 * dak
    al2 = (14 * eta2 - 4 * eta - 1) * dak2
    al3 = 36 * eta2 * dak4

    b1 = -(eta2 + 7. * eta + 1.) * dak
    b2 = 9. * eta * (eta2 + 4. * eta - 2.) * dak2
    b3 = 12. * eta * (2 * eta2 + 8. * eta - 1.) * dak4

    n1 = -(eta3 + 3. * eta2 + 45. * eta + 5.) * dak
    n2 = (eta32 + 3. * eta2 + 42. * eta - 20.) * dak2
    n3 = (eta32 + 30. * eta - 5.) * dak4
    n4 = n1 + 24. * eta * ak * n3
    n5 = eta6d * (n2 + 4. * n3)

    f1 = eta6d / ak
    f2 = d - 12. * eta * dak2

    ff1 = f1 * f1
    ff2 = f2 * f2
    ff = ff1 + ff2
    f1f2 = 2. * f1 * f2

    t1 = (eta + 5.) / (5. * ak)
    t2 = eta2d * dak2
    t3 = -12. * eta * gek * (t1 + t2)
    t4 = eta3d * ak2 * (t1 * t1 - t2 * t2)
    t5 = eta3d * (eta + 8.) * 0.1 - 2. * eta22 * dak2
    # ------------
    a1 = (e24 * gek * (al1 + al2 + ak1 * al3) - eta22) * dd4
    bb1 = (1.5 * eta * eta2d2 - 12. * eta * gek * (b1 + b2 + ak1 * b3)) * dd4
    v1 = (eta21 * (eta2 - 2. * eta + 10.) * 0.25 - gek * (n4 + n5)) * dd45
    p1 = (gek * (ff1 + ff2 - f1f2) - 0.5 * eta2d) * dd2
    T1 = t3 + t4 * a1 + t5 * bb1

    if (sak > 15) and (ix == 1):
        if debug > debuglevel: print('(sak>15) and (ix==1)', ak)
        # this corresponds to ibig=1 in original Hayter-Penfold code for large screening
        # large screening means the screening length 1/kappa is small compared to 2R and we are in the hard sphere limit
        # if ak is big -> cosh = sinh and a lot simplifies in asymptotic solution
        # but at same time cosh(ak) may exceeds numerical limits for really large ak
        a3 = e24 * (eta22 * dak2 - 0.5 * d2 - al3) * dd4
        bb3 = e12 * (0.5 * d2 * eta2d - eta3d * eta2d2 * dak2 + b3) * dd4
        v3 = ((eta3 - 6. * eta2 + 5.) * d - eta6d * (2. * eta3 - 3. * eta2 + 18. * eta + 10.) * dak2 + e24 * n3) * dd45
        p3 = (ff1 - ff2) * dd2
        T3 = t4 * a3 + t5 * bb3 + e12 * t2 - 0.4 * eta * (eta + 10.) - 1.
        M6 = T3 * a3 - e12 * v3 * v3
        M5 = T1 * a3 + a1 * T3 - e24 * v1 * v3
        M4 = T1 * a1 - e12 * v1 * v1
        L6 = e12 * p3 * p3
        L5 = e24 * p1 * p3 - 2. * bb3 - ak2
        L4 = e12 * p1 * p1 - 2. * bb1
        W56 = M5 * L6 - L5 * L6
        W46 = M4 * L6 - L4 * M6
        fa = -W46 / W56
        ca = -fa
        f = fa
        c = ca
        b = bb1 + bb3 * fa
        a = a1 + a3 * fa
        v = v1 + v3 * fa
        g1 = -(p1 + p3 * fa)
        fval = g1 if g1 > 1e-3 else 0.
        seta = evar
        # g24 = e24*gek*math.exp(ak)            # prevent math range error in exp for large ak (-> small scl)
        # u = (ak2*ak*ca-g24)/(ak2*g24)         # so we rewrite this to have exp(-ak)
        u = ak * ca / e24 / gek * math.exp(-ak) - 1 / ak2  # same as above two lines but this prevents math range error
        return ix, ir, fval, evar, reta, rgek, rak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1

    # small sak for the remaining
    sk = math.sinh(ak)
    ck = math.cosh(ak)
    ckma = ck - 1. - ak * sk
    skma = sk - ak * ck
    a2 = e24 * (al3 * skma + al2 * sk - al1 * ck) * dd4
    a3 = e24 * (eta22 * dak2 - 0.5 * d2 + al3 * ckma - al1 * sk + al2 * ck) * dd4

    bb2 = e12 * (-b3 * skma - b2 * sk + b1 * ck) * dd4
    bb3 = e12 * (0.5 * d2 * eta2d - eta3d * eta2d2 * dak2 - b3 * ckma + b1 * sk - b2 * ck) * dd4

    v2 = (n4 * ck - n5 * sk) * dd45
    v3 = ((eta3 - 6. * eta2 + 5.) * d - eta6d * (
            2. * eta3 - 3. * eta2 + 18. * eta + 10.) * dak2 + e24 * n3 + n4 * sk - n5 * ck) * dd45
    # define...
    p2 = (ff * sk + f1f2 * ck) * dd2
    p3 = (ff * ck + f1f2 * sk + ff1 - ff2) * dd2

    T2 = t4 * a2 + t5 * bb2 + e12 * (t1 * ck - t2 * sk)
    T3 = t4 * a3 + t5 * bb3 + e12 * (t1 * sk - t2 * (ck - 1.)) - 0.4 * eta * (eta + 10.) - 1.

    M1 = T2 * a2 - e12 * v2 * v2
    M2 = T1 * a2 + T2 * a1 - e24 * v1 * v2
    M3 = T2 * a3 + T3 * a2 - e24 * v2 * v3
    M4 = T1 * a1 - e12 * v1 * v1
    M5 = T1 * a3 + T3 * a1 - e24 * v1 * v3
    M6 = T3 * a3 - e12 * v3 * v3

    # ix is defined from the _sqcoef
    #  large k or close to GILLAN CONDITION g1==0 as explained in [1]
    if ix == 1 or ix == 3:
        # YES - G(X=1+) = 0
        # COEFFICIENTS AND FUNCTION VALUE
        L1 = e12 * p2 * p2
        L2 = e24 * p1 * p2 - 2. * bb2
        L3 = e24 * p2 * p3
        L4 = e12 * p1 * p1 - 2. * bb1
        L5 = e24 * p1 * p3 - 2. * bb3 - ak2
        L6 = e12 * p3 * p3

        W16 = M1 * L6 - L1 * M6
        W15 = M1 * L5 - L1 * M5
        W14 = M1 * L4 - L1 * M4
        W13 = M1 * L3 - L1 * M3
        W12 = M1 * L2 - L1 * M2
        W26 = M2 * L6 - L2 * M6
        W25 = M2 * L5 - L2 * M5
        W24 = M2 * L4 - L2 * M4
        W36 = M3 * L6 - L3 * M6
        W35 = M3 * L5 - L3 * M5
        W34 = M3 * L4 - L3 * M4
        W32 = M3 * L2 - L3 * M2
        W46 = M4 * L6 - L4 * M6
        W56 = M5 * L6 - L5 * M6

        # QUARTIC COEFFICIENTS W(I)
        #  these are used in
        # fun = w0+(w1+(w2+(w3+w4*fa)*fa)*fa)*fa  =w4*fa**4+w3*fa**3+w2*fa**2+w1*fa+w0
        w4 = W16 * W16 - W13 * W36
        w3 = 2. * W16 * W15 - W13 * (W35 + W26) - W12 * W36
        w2 = W15 * W15 + 2. * W16 * W14 - W13 * (W34 + W25) - W12 * (W35 + W26)
        w1 = 2. * W15 * W14 - W13 * W24 - W12 * (W34 + W25)
        w0 = W14 * W14 - W12 * W24
        # now find root of fun
        if useHP:
            # this documents the original HayterPenfold algorithm as found in original fortran code
            # to find the correct root an estimate is used and refined by Newton method
            # fails eg for R=3.1 gam=1.1 eta=0.5 when scl 6.1999 -> 6,2 as sak changes over 1
            # or scl=1.37382379588 R=2.5 gam=5.1 eta=0.6 as the found root results in g(r<1)>0
            # reason: in Newton refining an arbitrary root is found
            if ix == 1:  # large screening
                # LARGE K estimate for the zero of Fwww
                fap = (W14 - W34 - W46) / (W12 - W15 + W35 - W26 + W56 - W32)
            else:  # ix=3  no large screening
                # ASSUME NOT TOO FAR FROM GILLAN CONDITION.
                # IF BOTH RGEK AND RAK ARE SMALL, USE P-W ESTIMATE.of the zero of Fwww
                g1 = 0.5 * eta2d * dd2 * math.exp(-gek)
                pg = p1 + g1
                ca = ak2 * pg + 2. * (bb3 * pg - bb1 * p3) + e12 * g1 * g1 * p3
                ca = -ca / (ak2 * p2 + 2. * (bb3 * p2 - bb2 * p3))
                fap2 = -(pg + p2 * ca) / p3
                if (gek > 0) and (sgek <= 2.0) and (sak <= 1.0):
                    # gek>0 as this is only for positive contact potentials
                    # this was introduced in the SASFIT conversion (C code)
                    e24g = e24 * gek * math.exp(ak)
                    pwk = math.sqrt(e24g)
                    qpw = (1. - math.sqrt(1. + 2. * d2 * d * pwk / eta22)) * eta21 / d
                    g1 = -qpw * qpw / e24 + 0.5 * eta2d * dd2
                pg = p1 + g1
                ca = ak2 * pg + 2. * (bb3 * pg - bb1 * p3) + e12 * g1 * g1 * p3
                ca = -ca / (ak2 * p2 + 2. * (bb3 * p2 - bb2 * p3))
                fap = -(pg + p2 * ca) / p3
                # print('PWEstimate',fap,fap2,( sgek<=2.0) and ( sak<=1.0))
            # now find a better estimate of the zero by Newton iteration
            # RB: this algorithm finds different roots dependent on sgek and sak
            # the roots are somehow arbitrary in the 4 possible ones,
            # the main time it is one of the two centered which make no
            # big jumps but the outer ones make large jumps in the result
            ii = 0
            while True:
                ii += 1
                if ii > itm:  # FAILED TO CONVERGE IN ITM ITERATIONS
                    ir = -2
                    return ix, ir, fval, evar, reta, rgek, rak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1
                fa = fap  # estimated zero pole of fun
                fun = w0 + (w1 + (w2 + (w3 + w4 * fa) * fa) * fa) * fa  # function to minimize
                fund = w1 + (2. * w2 + (3. * w3 + 4. * w4 * fa) * fa) * fa  # derivative of fun
                fap = fa - fun / fund  # new value as next estimate
                if fa == 0: continue  # fa is 0 if gek is zero
                delta = abs((fap - fa) / fa)  # difference
                if delta < acc: break
            # found one and use this zero
            ir = ir + ii
            fa = fap
            ca = -(W16 * fa * fa + W15 * fa + W14) / (W13 * fa + W12)
            g1 = -(p1 + p2 * ca + p3 * fa)
        else:
            # original idea from Hayter paper [1]_
            # take all roots and use the physical root with g(r/diameter<1)=0
            # in this code there is no difference between ix=1 or 3
            # The algorithm relies on computing the eigenvalues of the companion matrix
            x0 = np.roots(
                [w4, w3, w2, w1, w0])  # 114µs      slower than direct calculation, but this is not the bottle neck
            if np.all((x0.imag / x0.real) < 1e-3):
                # if the imaginary part of complex roots is small use also these
                # in some cases this is the correct solution in gr
                fa = x0.real
            else:
                fa = x0[np.isreal(x0)].real  # 6.5µs
            fa.sort()  # we have up to 4 real roots and each of the following has up to 4 values
            ca = -(W16 * fa * fa + W15 * fa + W14) / (W13 * fa + W12)
            g1 = -(p1 + p2 * ca + p3 * fa)
            b = bb1 + bb2 * ca + bb3 * fa
            a = a1 + a2 * ca + a3 * fa
            # choose the correct root by calculating g(r) (sin transform) and using the one with g(r<1)=0
            # here i choose explicitly 1-delta
            delta = 0.05
            nn = (2 ** 13 + 0)  # n number of points to get reliable fft
            dqr2 = np.r_[0, delta:nn * delta:delta]  # points to calculate S(dqr2)
            kk = 1 // delta  # index of last point smaller 1
            # calc the value of g(x) with x=1-delta=kk*delta  in equ.12 of[1]_
            gr1 = [delta * np.sum(
                (_SQMSA(dqr2, scal, eta, ak, gek, aa, bb, cca, ffa) - 1) * dqr2 * np.sin(kk * delta * dqr2))
                   for aa, bb, cca, ffa in zip(a, b, ca, fa)]
            grval = [1 + ggr / (12 * np.pi * eta * kk * delta) for ggr in gr1]
            if len(fa) == 0 or np.min(grval) > 0.1:
                # no real root found or not grval close to zero
                ir = -4
                return ix, ir, fval, evar, reta, rgek, rak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, \
                       g1.max() if np.size(g1) else g1
            # chose the one with grval close to zero
            chooseone = np.argmin(np.abs(grval))
            if debug > debuglevel:
                # this writes the calculated g(r) to a file for checking of g(r)
                rrr = 2 * np.pi * np.fft.rfftfreq(len(dqr2), d=delta)  # points in r domain from rfft r/diameter
                # doing sin transform with rfft results in minus in front of imag part
                # compared to equation 12 in HP paper [1]
                # delta* is to get correct integral
                # gr=[delta*np.fft.rfft((_SQMSA(dqr2,scal,eta,ak,gek,aa,bb,cca,ffa)-1)*dqr2).imag
                #                                            for aa,bb,cca,ffa in zip(a,b,ca,fa)]
                gr = [delta * scipy.fft.dst(_SQMSA(dqr2, scal, eta, ak, gek, aa, bb, cca, ffa) - 1) * dqr2
                      for aa, bb, cca, ffa in zip(a, b, ca, fa)]
                # [1:] to avoid rrr=zero
                gr = [1 - ggr[1:] / (12 * np.pi * eta * rrr[1:]) for ggr in gr]
                # choose one with minimum mean value g(r) for rrr<1 which should be zero
                # above we use only one value and choose smallest grval
                # here we choose the smallest mean value which is often not correct but here it is only for demo
                grval = [grr[rrr[1:] < 0.9][1:].mean() for grr in gr]
                print('grval  ', grval)
                temp = dL()
                for i, grr in enumerate(gr):
                    temp.append(np.c_[rrr[1:], grr].T)
                    temp[-1].choosen = chooseone
                    temp[-1].zero = fa[i]
                    temp[-1].g1 = g1[i]
                    temp[-1].legend = 'g(r<1)= %.3g' % (grval[i])
                temp.savetxt('testgr.dat')
                print('zeros,g1,choosen zero', fa, g1, chooseone)
            fa = fa[chooseone]
            ca = ca[chooseone]
            g1 = -(p1 + p2 * ca + p3 * fa)
            # end searching the root- recalculating final result------------------------
        fval = (g1 if abs(g1) > 1e-3 else 0.)
        seta = evar
        f = fa
        c = ca
        b = bb1 + bb2 * ca + bb3 * fa
        a = a1 + a2 * ca + a3 * fa
        v = (v1 + v2 * ca + v3 * fa) / a

    else:
        # -> ix==2 or ix==4
        ca = ak2 * p1 + 2. * (bb3 * p1 - bb1 * p3)
        ca = -ca / (ak2 * p2 + 2.0 * (bb3 * p2 - bb2 * p3))
        fa = -(p1 + p2 * ca) / p3
        # fval will contain g1 for Newton iteration ix=2,4
        if ix == 2:    fval = M1 * ca * ca + (M2 + M3 * fa) * ca + M4 + M5 * fa + M6 * fa * fa
        if ix == 4:    fval = -(p1 + p2 * ca + p3 * fa)
        f = fa
        c = ca
        b = bb1 + bb2 * ca + bb3 * fa
        a = a1 + a2 * ca + a3 * fa
        v = (v1 + v2 * ca + v3 * fa) / a
    g24 = e24 * gek * math.exp(ak)
    u = (ak2 * ak * ca - g24) / (ak2 * g24)
    return ix, ir, fval, evar, reta, rgek, rak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1


def _SQMSA(qR2, scal, eta, ak, gek, a, b, c, f):
    """
    equation 14 Hayter-Penfold paper [1] in sfRMSA to calculate the final structure factor
    """
    K = np.where(qR2 == 0, 1e-15, qR2 / scal)  # catch zero
    if ak > 25:  # c==-f and
        # avoid to large ak to prevent math range error
        # ak>15 has f=-c from sqfun so the sinh and cosh terms cancel for large ak
        sinhsk = 0.
        coshsk = 0.
    else:
        sinhsk = math.sinh(ak)
        coshsk = math.cosh(ak)
    sink = np.sin(K)
    cosk = np.cos(K)
    K2 = K * K
    K3 = K2 * K
    K4 = K3 * K
    KK2ak2 = 1 / K / (K2 + ak ** 2)
    a_K = a * (sink - K * cosk) / K3 \
          + b * ((2. / K ** 2 - 1) * K * cosk + 2 * sink - 2. / K) / K3 \
          + a * eta * (24. / K3 + 4. * (1 - 6. / K2) * sink - (1 - 12. / K2 + 24. / K4) * K * cosk) / 2. / K3 \
          + c * (ak * coshsk * sink - K * sinhsk * cosk) * KK2ak2 \
          + f * (ak * sinhsk * sink - K * (coshsk * cosk - 1)) * KK2ak2 \
          + f * (cosk - 1) / K2 \
          - gek * (ak * sink + K * cosk) * KK2ak2
    msa = 1 / (1 - 24. * eta * a_K)
    MSA = np.where(qR2 == 0, -1 / a, msa)  # -1/a is correct solution for qR2==zero
    return MSA

def RMSA(q, R, scl, gamma, molarity=None, eta=None, useHP=False):
    r"""
    Structure factor for a screened coulomb interaction (single Yukawa) in rescaled mean spherical approximation (RMSA).

    Structure factor according to Hayter-Penfold [1]_ [2]_ .
    Consider a scattering system consisting of macro ions, counter ions  and solvent.
    Here an improved algorithm [3]_ is used based on the original idea described in [1]_ (see Notes).

    Parameters
    ----------
    q : array; N dim
        Scattering vector; units 1/nm
    R : float
        Radius of the object; units nm
    molarity : float
        Number density n in units mol/l. Overrides eta, if both given.
    scl : float>0
        Screening length; units nm; negative values evaluate to scl=0.
    gamma : float
        Contact potential :math:`\gamma` in units kT.
         - :math:`\gamma=Z_m/(\pi \epsilon \epsilon_0 R (2+\kappa R))`
          - :math:`Z_m = Z^*` effective surface charge
          - :math:`\epsilon_0,\epsilon` free permittivity and dielectric constant
          - :math:`\kappa=1/scl` inverse screening length of Debye-Hückel potential
    eta : float
        Volume fraction as eta=:math:`4/3piR^3n`  with number density n.
    useHP : True, default False
        To use the original Hayter/Penfold algorithm. This gives wrong results for some parameter conditions.
        It should ONLY be used for testing.
        See example examples/test_newRMSAAlgorithm.py for a direct comparison.

    Returns
    -------
    dataArray
         - .volumeFraction = eta
         - .rescaledVolumeFraction
         - .screeningLength
         - .gamma=gamma
         - .contactpotential
         - .S0 structure factor at q=0
         - .scalingfactor factor for rescaling to get g+1=0; if =1 nothing was scaled and it is MSA

    Notes
    -----
    The repulsive potential between two identical spherical macroions of diameter :math:`\sigma` is (DLVO model)
    in dimensionless form

    .. math:: \frac{U(x)}{k_BT} = \gamma \frac{e^{-kx}}{x}   \; for \; x>1

    - :math:`x = r/\sigma, k=\kappa\sigma, K=Q\sigma`
    - :math:`k_BT` thermal energy
    - :math:`\gamma e^{-k} = \frac{\pi \epsilon_0 \epsilon \sigma }{k_BT} \psi^2_0` contact potential in kT units
    - The potential is completed by :math:`U(x)/kT=\infty , x<1`

    - From [1]_:
       This potential is valid for colloid systems provided k < 6.
       There is no theoretical restriction on k in what follows, however, and for general studies
       of one component plasmas any value may be used.
    - In the limit :math:`\gamma \rightarrow 0` or :math:`k\rightarrow\infty` the Percus-Yevick hard sphere is reached.
    - Why is is named **rescaled MSA**:
      From [1]_:
       Note that in general, however, the MSA fails at low density; letting :math:`n\rightarrow0` yields
       :math:`g(x)\rightarrow 1-lU(x)/kT` for x> 1. Since U(x) is generally larger than thermal energies
       for small interparticle separations, g(x) will generally be negative (and hence unphysical)
       near the particle at very low densities.
       This does not present a problem for many colloid studies of current interest, where volume fractions are
       generally greater than 1%.

      To solve this the radius is rescaled to get :math:`g(\sigma +)=0` according to [2]:
        ...by increasing the particle diameter from its physical value `a` to an effective hard core value `a'`,
        while maintaining the Coulomb coupling constant. ...

      If :math:`g(\sigma +)>=0` no rescaling is done.


    Improved algorithm (see [3]_ fig. 6)
     The Python code is deduced from the original Hayter-Penfold Fortran code (1981, ILL Grenoble).
     This is also used in other common SAS programs as SASfit or SASview (translated to C).
     The original algorithm determines the root of a quartic F(w1,w2,w3,w4) by an estimate (named PW estimate),
     refining it by a Newton algorithm. As the PW estimate is sometimes not good enough this results in an
     arbitrary root of the quartic in the Newton algorithm. The solution therefore jumps between different
     possibilities by small changes of the parameters.
     We use here the original idea from [1]_ to calculate G(r<0) for all four roots of F(w1,w2,w3,w4) and use
     the physical solution with G(r<R)=0.
     See examples/test_newRMSAAlgorithm.py for a direct comparison or [3]_ fig. 6.

    Validity
     The calculation of charge at the surface or screening length from a solute ion concentration is explicitly dedicate
     to the user. The Debye-Hückel theory for a macro ion in screened solution is a far field theory as a linearization
     of the Poisson-Boltzmann (PB) theory and from limited validity (far field or low charge -> linearization).
     Things like reverting charge layer, ion condensation at the surface, pH changes at the surface or other things
     might appear. Before calculating please take these things into account. Close to the surface the PB
     has to be solved. The DH theory can still be used if the charge is thus an effective charge named Z*,
     which might be different from the real surface charge.
     See Ref [4]_ for details.

    Examples
    --------
    Effect of volume fraction, surface potential and screening length onto RMSA structure factor
    ::

     import jscatter as js
     R = 6
     eta0 = 0.2
     gamma0 = 30 # surface potential
     scl0 = 10
     q = js.loglist(0.01, 5, 200)
     p = js.grace(1,1.5)
     p.multi(3,1)
     for eta in [0.01,0.05,0.1,0.2,0.3,0.4]:
         rmsa = js.sf.RMSA(q, R, scl=scl0, gamma=gamma0, eta=eta)
         p[0].plot(rmsa, symbol=0, line=[1, 3, -1], legend=f'eta ={eta:.1f}')
     for scl in [0.1,1,5,10,20]:
         rmsa = js.sf.RMSA(q, R, scl=scl, gamma=gamma0, eta=eta0)
         p[1].plot(rmsa, symbol=0, line=[1, 3, -1], legend=f'scl ={scl:.1f}')
     for gamma in [1,10,20,40,100]:
         rmsa = js.sf.RMSA(q, R, scl=scl0, gamma=gamma, eta=eta0)
         p[2].plot(rmsa, symbol=0, line=[1, 3, -1], legend=r'\xG\f{} =$gamma')
     p[0].yaxis(min=0.0, max=2.5, label='S(Q)', charsize=1.5)
     p[0].legend(x=1.2, y=2.4)
     p[0].xaxis(min=0, max=1.5,label='')
     p[1].yaxis(min=0.0, max=2.2, label='S(Q)', charsize=1.5)
     p[1].legend(x=1.1, y=2.)
     p[1].xaxis(min=0, max=1.5, label=r'')
     p[2].yaxis(min=0.0, max=2.2, label='S(Q)', charsize=1.5)
     p[2].legend(x=1.1, y=2.2)
     p[2].xaxis(min=0, max=1.5, label=r'Q / nm\S-1')
     p[0].title('RMSA structure factor')
     p[0].subtitle(f'R={R:.1f} gamma={gamma0:.1f} eta={eta0:.2f} scl={scl0:.2f}')
     #p.save(js.examples.imagepath+'/rmsa.jpg',size=[600,900])

    .. image:: ../../examples/images/rmsa.jpg
     :width: 50 %
     :align: center
     :alt: rmsa

    References
    ----------
    .. [1] J. B. Hayter and J. Penfold, Mol. Phys. 42, 109 (1981).
    .. [2] J.-P. Hansen and J. B. Hayter, Mol. Phys. 46, 651 (2006).
    .. [3] Jscatter, a program for evaluation and analysis of experimental data
           R.Biehl, PLOS ONE, 14(6), e0218789, 2019,  https://doi.org/10.1371/journal.pone.0218789
    .. [4] L. Belloni, J. Phys. Condens. Matter 12, R549 (2000).

    """

    """
    Original Doc of the Hayter Penfold Fortran routine::
        
    seta is the rescaled volume fraction.                             
    sgek is the rescaled contact potential.                           
    sak is the rescaled screening constant.                           
    a,b,c,f,u,v are the msa coefficients.                             
    g1=g(1+) is the contact value of g(r/sig);                        
    for the Gillan condition, the difference from                     
    zero indicates the computational accuracy.                        

      ROUTINE TO CALCULATE S(Q*SIG) FOR A SCREENED COULOMB
      POTENTIAL BETWEEN FINITE PARTICLES OF DIAMETER 'SIG'
      AT ANY VOLUME FRACTION. THIS ROUTINE IS MUCH MORE POWER-
      FUL THAN "SQHP" AND SHOULD BE USED TO REPLACE THE LATTER
      IN EXISTING PROGRAMS. NOTE THAT THE COMMON AREA IS
      CHANGED; IN PARTICULAR, THE POTENTIAL IS PASSED
      DIRECTLY AS 'GEK' = GAMMA*EXP(-K) IN THE PRESENT ROUTINE.
      JOHN B.HAYTER (I.L.L.) 19-AUG-81
 
      CALLING SEQUENCE:
       CALL SQHPA(QQ,SQ,NPT,IERR)
 
      QQ: ARRAY OF DIMENSION NPT CONTAINING THE VALUES  OF Q*SIG AT WHICH S(Q*SIG) WILL BE CALCULATED.
      SQ: ARRAY OF DIMENSION NPT INTO WHICH VALUES OF  S(Q*SIG) WILL BE RETURNED.
      NPT: NUMBER OF VALUES OF Q*SIG.
 
      IERR > 0: NORMAL EXIT; IERR=NUMBER OF ITERATIONS.
       -1: NEWTON ITERATION NON-CONVERGENT IN "SQCOEF"
       -2: NEWTON ITERATION NON-CONVERGENT IN "SQFUN".
       -3: CANNOT RESCALE TO G(1+) > 0.
 
      ON ENTRY:
      ETA: VOLUME FRACTION
      GEK: THE CONTACT POTENTIAL GAMMA*EXP(-K)
      AK: THE DIMENSIONLESS SCREENING CONSTANT
      AK = KAPPA*SIG WHERE KAPPA IS THE INVERSE SCREENING
      LENGTH AND SIG IS THE PARTICLE DIAMETER.
 
      ON EXIT:
      GAMK IS THE COUPLING: 2*GAMMA*S*EXP(-K/S), S=ETA**(1/3).
      SETA, SGEK AND SAK ARE THE RESCALED INPUT PARAMETERS.
      SCAL IS THE RESCALING FACTOR: (ETA/SETA)**(1/3).
      G1=G(1+), THE CONTACT VALUE OF G(R/SIG).
      A,B,C,F,U,V ARE THE CONSTANTS APPEARING IN THE ANALYTIC
      SOLUTION OF THE MSA (HAYTER-PENFOLD; MOL.PHYS. 42: 109 (1981))
 
      NOTES:
      (A) AFTER THE FIRST CALL TO SQHPA, S(Q*SIG) MAY BE EVALUATED
      AT OTHER Q*SIG VALUES BY REDEFINING THE ARRAY QQ AND CALLING
      "SQHCAL" DIRECTLY FROM THE MAIN PROGRAM.
      (B) THE RESULTING S(Q*SIG) MAY BE TRANSFORMED TO G(R/SIG)
      USING THE ROUTINE "TROGS".
      (C) NO ERROR CHECKING OF INPUT PARAMETERS IS PERFORMED;
      IT IS THE RESPONSIBILITY OF THE CALLING PROGRAM TO VERIFY
      VALIDITY.
      SUBROUTINES REQUIRED BY SQHPA:
      (1) SQCOEF RESCALES THE PROBLEM AND CALCULATES THE
       APPROPRIATE COEFFICIENTS FOR "SQHCAL".
      (2) SQFUN CALCULATES VARIOUS VALUES FOR "SQCOEF".
      (3) SQHCAL CALCULATES H-P S(Q*SIG) GIVEN A,B,C,F.


    """
    R = abs(R)
    error = {-1: 'NEWTON ITERATION NON-CONVERGENT IN _sqcoef',
             -2: 'NEWTON ITERATION NON-CONVERGENT IN _sqfun',
             -3: 'CANNOT RESCALE TO G(1+) > 0.',
             -4: 'no physical root with G(r<1) < 0.1 in _sqfun found'}  # added for new algorithm
    # get volume fraction eta from number density and radius R
    if isinstance(molarity, numbers.Number):
        molarity = abs(molarity)
        numdens = constants.N_A * molarity * 1e-24  # from mol/l to particles/nm**3
        eta = 4 / 3. * np.pi * R ** 3 * numdens
    elif isinstance(eta, numbers.Number):
        numdens = eta / (4 / 3. * np.pi * R ** 3)
        molarity = numdens / (constants.N_A * 1e-24)
    else:
        raise Exception('one of molarity/eta needs to be given.')  # dimensionless screening constant ak
    if eta <= 0.: eta = 1e-10
    # if eta>1:        raise Exception('eta needs to be smaller 1.')
    if scl <= 0:
        ak = 1e20
    else:
        ak = 2 * R / scl
    # to large ak make math error in exp , anyway then we have a hard sphere
    if ak > 200: ak = 200
    # the contact potential in kT
    gek = gamma * math.exp(-ak)
    # coupling
    gamk = 2. * eta ** (1 / 3.) * gek * math.exp(ak - ak / eta ** (1 / 3.))
    # ----------do the rescaling in _sqcoef--------------------
    # _sqcoef does the rescaling to satisfy the Gillian condition with g1==0 according to [2]
    # therein _sqfun calculates the NOT rescaled solution described in [1]
    if useHP:
        ir, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1 = _sqcoefOriginalHP(ir=0, eta=eta, gek=gek,
                                                                                                ak=ak, gamk=gamk)
    else:
        ir, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1 = _sqcoef(ir=0, eta=eta, gek=gek, ak=ak,
                                                                                      gamk=gamk)
    # catch error
    if ir < 0:
        print(ir, error[ir], 'g+ =', g1, 'ak=', ak)
        return ir

    # dimensionless q scale
    q = np.atleast_1d(q)
    qR2 = 2 * R * q
    # calc values by _SQMSA
    sq = _SQMSA(qR2, scal, seta, sak, sgek, a, b, c, f)
#     result = dA(np.r_[[q, sq]])
#     result.setColumnIndex(iey=None)
#     # add important parameters
#     result.volumeFraction = eta
#     result.rescaledVolumeFraction = seta
#     result.molarity = molarity
#     result.screeningLength = scl
#     result.gamma = gamma
#     result.contactpotential = gek
#     result.S0 = -1 / a
#     result.scalingfactor = scal
#     result.gplus1 = [g1, ir]
#     result.modelname = inspect.currentframe().f_code.co_name
#     result._coefficients = {key: value for (value, key) in
#                             zip([ir, eta, gek, ak, a, b, c, f, u, v, gamk, seta, sgek, sak, scal, g1],
#                                 ['ir', 'eta', 'gek', 'ak', 'a', 'b', 'c', 'f', 'u', 'v', 'gamk', 'seta', 'sgek', 'sak',
#                                  'scal', 'g1'])}
    return q, sq