def transverse_resonator(f: float, fr: float, Rperp: float, Q:float, beta:float) -> complex:
    '''
    Ideal transverse (m=1) resonator impedance
    
    Z⟂(ω) = ωr/ω * R⟂ / (1 + i*Q*(ωr/ω - ω/ωr)),

    where ω is the angular frequency, ωr is the resonant angular frequency,
    and R⟂ is the transverse resistance in units of Ohm/m [1]. Note that this is
    equal to Chao's Equation 2.87, with the substitution
    R⟂ = c/omega_r * Rs(m=1), where Rs is the shunt impedance as defined by
    Chao [2]. R⟂ has units of Ohm/m whilst Rs(m=1) has units of Ohm/m^2.


    **References**

    [1] E. Métral and M. Migliorati, ‘Longitudinal and transverse mode
    coupling instability: Vlasov solvers and tracking codes’, Physical Review
    Accelerators and Beams, vol. 23, Jul. 2020, doi:
    10.1103/PhysRevAccelBeams.23.071001.

    [2] A. W. Chao, Physics of Collective Beam Instabilities in High Energy
    Accelerators, 1st ed. John Wiley & Sons, 1993. [Online]. Available:
    http://www.slac.stanford.edu/%7Eachao/wileybook.html
    '''

    return (fr / f) * Rperp / (1 + 1j * Q * ((fr / f) - (f / fr)))
