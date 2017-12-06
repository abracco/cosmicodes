pro polmodes, Q, U, E, B, ef, bf

;------------------------------------------------------------
; polmodes.pro
;
; The routine produces E and B polarization mode maps on a flat sky,
; given the Stokes Q and U maps. The latter should have periodic
; boundary conditions and possibly no point sources. Apodization is
; not introduced yet.      
;  
; INPUTS
;
; Q,U - Stokes parameter input maps
;
; OUTPUTS
;
; E,B - E and B maps
; ef, bf - complex numbers corresponding to the Fourier
;          transform of E and B (needed to compute power spectra)
;
; mamdlib needed for xymap.pro
;
; Andrea Bracco, andrea.bracco@su.se
;------------------------------------------------------------
  
sz=size(Q) 

xymap,sz(1),sz(2),kx,ky
kx1=fix(kx-sz(1)/2.)
ky1=fix(ky-sz(2)/2.)
k=sqrt(kx1^2.+ky1^2.)
k(where(k eq 0))=0.01

sn=(2*kx1*ky1)/(k*k)
cs=(kx1^2.-ky1^2.)/(k*k)

sf=complex(sn,0)
cf=complex(cs,0)

qf=fft(Q,-1,/center)
uf=fft(U,-1,/center)
ef=qf*cf + uf*(sf)
bf=-qf*(sf) + uf*cf

E=real_part((fft((ef),/center,/inverse)))
B=real_part((fft((bf),/center,/inverse)))

end

