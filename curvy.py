# find curvature, orientation, and gradient of a 2D map with Gaussian derivatives
# Input: 2D map and sigma in pixels of the desired Gaussian.

def curvy(map,sigma):

    sz_in=np.shape(map)

    x = np.linspace(0, sz_in[0]-1, sz_in[0])
    y = np.linspace(0, sz_in[1]-1, sz_in[1])
    xx, yy = np.meshgrid(x, y , indexing='ij')

    r=np.sqrt((xx-sz_in[0]/2.)**2+(yy-sz_in[1]/2.)**2)
    gauss=(1./(sigma*np.sqrt(2.*math.pi)))*np.exp(-0.5*(r/sigma)**2.)
    gaussn=gauss/np.sum(gauss)

    imf = np.fft.fftshift(np.fft.fft2(map))
    mhf = np.fft.fftshift(np.fft.fft2(gaussn))

    nx = sz_in[0]
    ny = sz_in[1]

    xmap = xx -nx/2.
    ymap = yy -ny/2.

    # first derivatives

    it1x=2*math.pi*1j/nx*xmap*imf
    it1y=2*math.pi*1j/ny*ymap*imf

    # smoothed input map and maps of x-y derivatives
    
    ims=np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(imf*mhf))))
    dx = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(it1x*mhf))))
    dy = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(it1y*mhf))))

    #  second derivatives
    it1xx=-(2*math.pi*xmap/nx)**2*imf
    it1yy=-(2*math.pi*ymap/ny)**2*imf
    it1xy=-(2*math.pi)**2*xmap*ymap/(nx*ny)*imf

    dxxf = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(it1xx*mhf))))
    dyyf = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(it1yy*mhf))))
    dxyf = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(it1xy*mhf))))

    #  eigenvalues of the hessian matrix, negative and positive curvature, orientation.
    c1f=0.5*(dxxf+dyyf-np.sqrt((dxxf-dyyf)**2+4*dxyf**2))
    c2f=0.5*(dxxf+dyyf+np.sqrt((dxxf-dyyf)**2+4*dxyf**2))
    thetaf=0.5*np.arctan2(2*dxyf,(dxxf-dyyf))

    return c1f,c2f,thetaf,np.sqrt(dx**2+dy**2),dx,dy,dxxf,dyyf,dxyf,ims

