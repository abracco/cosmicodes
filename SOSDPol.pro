pro SOSDPol, NSIDE, fM, N, alphaM, l0, b0, resol, Qm, Um

;------------------------------------------------------------
; SOS-DPol: Simulations On the Sphere of Dust Polarization
;
; The routine produces modelled Stokes parameters for linear polarization Q and U accounting for the Galactic magnetic field structure only (Planck Int. results. 2016. XLIV). IDL and HEALPix are necessary.   
; 
; INPUTS
;
; NSIDE - HEALPix determines the pixel resolution on the sphere
; fM -  turbulent-to-ordered field ratio (equipartition case fM=1,
;      strong turbulent field fM > 1.)
; N - number of uncorrelated polarization layers along the line of
;     sight (same ordered field but independent turbulent component);
;     if N=0 then default N is changed to 1,
; alphaM - spectral index of the power-law power spectrum used to model the turbulent field (Gaussian fluctuations). This is the same for all N-polarization layers.
; l0 and b0 - Galactic coordinates in degrees for the direction of the ordered component of the field (i.e., l0=70 b0=24)
; resol - angular resolution in arcmin of the output maps
;
; OUTPUTS
;
; Qm and Um - correspond to the modelled Stokes parameters
;
; Andrea Bracco, andrea.bracco@su.se, abracco@irb.hr 
;------------------------------------------------------------

if n_params() eq 0 then begin
print," "
print,"CALLING SEQUENCE:"
print,"SOSDPol, NSIDE, fM, N, alphaM, l0, b0, resol, Qm, Um"
print," "
return,0
endif  

; Setting number of layers to 1 if the input is 0
if N eq 0 then N=1 
  
; Building the Field structure: turbulent power spectrum
el=findgen(nside*3+1)
cl=el^(alphaM)
cl(0:1)=0.

;
; LOS vectors
;

npix=nside2npix(nside)
lpix=lindgen(npix)
pix2vec_ring, nside, lpix, los_vec

;
; North and East vectors
;

pix2ang_ring,nside,lpix,theta,phi
glat=(!pi/2-theta)*!radeg
north_vec=dblarr(npix,3)
east_vec=dblarr(npix,3)
for k=0L,npix-1L do north_vec[k,*] = [-cos(phi[k])*sin(!pi/2.-theta[k]),-sin(phi[k])*sin(!pi/2.-theta[k]),cos(!pi/2.-theta[k])]
for k=0L,npix-1L do east_vec[k,*] = -[los_vec[k,1]*north_vec[k,2]-los_vec[k,2]*north_vec[k,1], $
                                                      -los_vec[k,0]*north_vec[k,2]+los_vec[k,2]*north_vec[k,0], $
                                                       los_vec[k,0]*north_vec[k,1]-los_vec[k,1]*north_vec[k,0]]

;
; large scale field magnetic field
;

lon_b0=l0*1./!radeg & lat_b0=b0*1./!radeg

B0vec=[cos(lon_b0)*cos(lat_b0),sin(lon_b0)*cos(lat_b0),sin(lat_b0)]

;
; Sky map of B field
;

Bvec=fltarr(npix,3)
Q_cube=dblarr(npix,N)
U_cube=dblarr(npix,N)

for j=0, N-1 do begin & $

       isynfast,cl,mapx,nside=nside,lmax=2*nside+1,fwhm_arcmin=resol,iseed=1000+j,/silent,simul_type=1
       isynfast,cl,mapy,nside=nside,lmax=2*nside+1,fwhm_arcmin=resol,iseed=2000+j,/silent,simul_type=1
       isynfast,cl,mapz,nside=nside,lmax=2*nside+1,fwhm_arcmin=resol,iseed=3000+j,/silent,simul_type=1

       sig_map=mean([stdev(mapx),stdev(mapy),stdev(mapz)])
       fac=fM/(sig_map*sqrt(3.)) ; stdev(Bturb)/B0

       for i=0l, n_elements(lpix)-1 do begin & $
   
	      Bvec[i,0]=B0vec[0]+mapx[i]*fac
   	      Bvec[i,1]=B0vec[1]+mapy[i]*fac
   	      Bvec[i,2]=B0vec[2]+mapz[i]*fac
    
       endfor

; normalization to unit length
       for k=0L,npix-1L do Bvec[k,*]=Bvec[k,*]/sqrt(Bvec[k,0]^2.+Bvec[k,1]^2.+Bvec[k,2]^2.)

;
; compute maps of Stokes parameters
;

	Bvec_perp=fltarr(npix,3)
 	B_times_los=fltarr(npix) ; scalar producet B.los
 	for k=0L,npix-1L do B_times_los[k]=total(Bvec[k,*]*los_vec[k,*])
	Bvec_perp[*,0]=Bvec[*,0]-los_vec[*,0]*B_times_los
        Bvec_perp[*,1]=Bvec[*,1]-los_vec[*,1]*B_times_los
 	Bvec_perp[*,2]=Bvec[*,2]-los_vec[*,2]*B_times_los

 	P_I0=(Bvec_perp[*,0]^2.+Bvec_perp[*,1]^2.+Bvec_perp[*,2]^2.)
;
; compute Q anbd U maps
;

        B_angle=fltarr(npix)
 	Qmap=fltarr(npix) & Umap=fltarr(npix)
        g=where(P_I0 gt 0.,ng)
 	buf = (Bvec_perp[g,0]*north_vec[g,0]+Bvec_perp[g,1]*north_vec[g,1]+Bvec_perp[g,2]*north_vec[g,2])/sqrt(P_I0[g])
 	b = where(buf gt 1.,nb)   &  if(nb gt 0) then buf[b] = 1.
        b = where(buf lt -1.,nb)  &  if(nb gt 0) then buf[b] = -1.
        B_angle[g] = acos(buf)        ; from 0 to !pi
        buf =  (Bvec_perp[*,0]*east_vec[*,0]+Bvec_perp[*,1]*east_vec[*,1]+Bvec_perp[*,2]*east_vec[*,2])
        neg_angle = where(buf lt 0.)
 	B_angle[neg_angle] = -B_angle[neg_angle]
 	pol_angle = B_angle + !pi/2.
 	b=where(pol_angle gt !pi)
 	pol_angle[b]=  pol_angle[b]-2.*!pi
 	Qmap = P_I0*cos(2.*pol_angle)
 	Umap = -P_I0*sin(2.*pol_angle)   ; minus sign Planck vs IAU convention

 	psi_pol=pol_angle ; polarisation angle from -pi/2 to !pi/2
 	bn=where(psi_pol lt -!pi/2.)
 	psi_pol[bn]=psi_pol[bn]+!pi
 	bp=where(psi_pol gt !pi/2.)
 	psi_pol[bp]=psi_pol[bp]-!pi

	Q_cube[*,j]=qmap;*randomu(4000+j,npix)
 	U_cube[*,j]=umap;*randomu(5000+j,npix)

        Bvec[*,*]=0.

endfor

; Preparing Outputs

Qm=q_cube[*,0]*0.
Um=Qm*0.

for j=0,N-1 do begin
   Qm[*]+=q_cube[*,j]
   Um[*]+=u_cube[*,j]
endfor

Qm/=(1.*N)
Um/=(1.*N)

end


