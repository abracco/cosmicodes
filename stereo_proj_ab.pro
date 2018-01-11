pro stereo_proj_ab, nside, nx, ny, radius, m_in, m_out, l0, b0, l1, b1,ncont

;---------------------------------------
; This routine produces a stereographic projection (m_out) of the HEALPix map given as input (m_in). The code is adapted from http://mathworld.wolfram.co;m/StereographicProjection.html
;
; INPUTS
;
; nside - is the HEALPix parameter of m_in
; nx, ny, - are the x,y dimensions of the output map
; radius - is the radius in degrees of the selected area that is projected
; m_in - input map
; l0 - the central Galactic longitude in degrees
; b0 - the central Galactic latitude in degrees
; ncont - number of coordinates contours in the final image
;
; OUTPUTS
;
; m_out - output map nx times ny
; l1 and b1 - output maps of projected Galactic coordinates in radians
;
; NOTES : to run the script IDL, HEALPix, and mamdlib (https://www.ias.u-psud.fr/pperso/mmiville/mamdlib.html) are needed
;
; Example:
; 
; IDL > stereo_proj_ab,2048,500,500,40.,m_in,m_out,190.,-27.,lon1,lat1,6
;  
; the routine takes a circular surface of radius 40 degrees of a
; HEALPix map (m_in) with NSIDE=2048 centered in (190,-27) and reproject it
; on a 500x500 stereographic map (m_out) with 6 contours for l and b,
; which are also saved in distinct maps (lon1 and lat1).  
;-------------------------------------------------  
; Andrea Bracco, andrea.bracco@cea.fr, April 2017
;-------------------------------------------------

npix=nside2npix(nside)
lpix=lindgen(npix)
pix2ang_ring,nside,lpix,theta,phi
glon=phi
glat=!pi/2.-theta

l0=l0/!radeg
b0=b0/!radeg
vec_tmp=[cos(l0)*cos(b0),sin(l0)*cos(b0),sin(b0)]
sel=where(finite(glon) ne 0)
if radius ne 0 then query_disc,nside,vec_tmp, radius, sel, /deg
mask=m_in*0
mask(sel)=1
orthview,m_in*mask,rot=[l0*!radeg,b0*!radeg],/half_sky,/log,grat=30,glsize=1.3,title='Orthographic test projection'

lr=glon(sel)
br=glat(sel)

R=1
A=2.*R/(1+sin(b0)*sin(br)+cos(b0)*cos(br)*cos(lr-l0))
sel2=where(A lt 1e4)
x1= A*cos(br)*sin(lr-l0)
y1= A*(cos(b0)*sin(br)-sin(b0)*cos(br)*cos(lr-l0))
x1=x1(sel2)
y1=y1(sel2)

xymap,nx,ny,xnn,ynn
xnn=-(xnn/(nx-1.)*(max(x1)-min(x1))+min(x1))
ynn=ynn/(ny-1.)*(max(y1)-min(y1))+min(y1)

rho=sqrt(xnn^2+ynn^2)
c=2*atan(rho/(2.*R))

b1=asin(cos(c)*sin(b0)+(ynn*sin(c)*cos(b0))/(rho))
l1=l0+atan((xnn*sin(c)),(rho*cos(b0)*cos(c)-ynn*sin(b0)*sin(c)))

sel1=where(finite(l1) ne 0)

m_out=l1*0.

for j=0,n_elements(sel1)-1 do begin & $

     ang2pix_ring,nside,!pi/2-b1(j),l1(j),indp & $
     m_out(j)=m_in(indp) & $

endfor

   window,2,xsize=600,ysize=600 & imaffi, alog10(m_out),imrange=[-2,1],title='Stereographic projection, alog10(M_in)',/bar,xtitle='pixels',ytitle='pixels'
ncont=5
lp1=gl1*!radeg
bp1=gb1*!radeg

levl=findgen(ncont)/(ncont-1.)*(max(lp1)-min(lp1))+min(lp1)
levb=findgen(ncont)/(ncont-1.)*(max(bp1)-min(bp1))+min(bp1)

contour,lp1,lev=levl,c_lab=string(levl),/overplot
contour,bp1,lev=levb,c_lab=string(levb),/overplot


end
