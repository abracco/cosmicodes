function qu_coord_transform,nside,mq,mu,coordin,coordout

; Rotate Q and U maps from two different coordinate systems

if n_params() eq 0 then begin
print," "
print,"CALLING SEQUENCE:"
print,"QU_out = qu_coord_transform, nside, mq, mu,coordin,coordout"
print," "
return,0
endif  


npix=nside2npix(nside)
lpix=lindgen(npix)
pix2ang_ring,nside,lpix,theta,phi
glon=phi
glat=!pi/2.-theta

vec=[[cos(glon)*cos(glat)],[sin(glon)*cos(glat)],[sin(glat)]]
cbpol=[[mq],[mu]]
cbout=rotate_coord(vec,inco=coordin,outco=coordout,Stokes_parameters=cbpol)

return, cbpol

end


