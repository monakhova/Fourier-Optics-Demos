# Functions 
def rgb2gray(rgb):
    import numpy as np   
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def FresProp(img,L,lmda,z):
    import numpy as np   
    [M, N] = np.shape(img)    # Input field array size 
    dx = L/M                 # Sample interval 
    fx = np.arange(-1/(2*dx), 1/(2*dx), 1/L)  # frequency coordinates
    fy = fx
    [FX, FY] = np.meshgrid(fx,fy)
    
    H = np.exp(-1j*np.pi*lmda*z*((FX)**2 + (FY)**2))  # Tranfer Function 
    H = fft.fftshift(H)                            # Shifter TF
    
    U1 = fft.fft2(fft.fftshift(img))
    U2 = H*U1
    u2 = fft.ifftshift(fft.ifft2(U2))
    
    return(u2, H)

def contrast(g_in):
    import numpy as np   
    I_max = np.max(np.abs(g_in)**2)
    I_min = np.min(np.abs(g_in)**2)
    C = (I_max - I_min)/(I_max+I_min)
    return(C)
    

def propogate(ein, lmda, z, ps):
    import numpy as np   
    # Digitally refocuses a complex field a given distance, z. 
    # (ref pg 67,J Goodman, Introduction to Fourier Optics)
    #inputs:
    #         ein    - complex field at input plane
    #         lmda   - wavelength of light [um]
    #         z      - vector of propagation distances [um]
    #         ps     - pixel size [um]
    [m,n]=ein.shape; M = m; N= n;
    
    eout = np.zeros([m,n,z.shape[0]])*1j;
    f_metric = np.zeros([z.shape[0]])*1j;
    Hout = np.zeros([m,n,z.shape[0]])*1j;
    
    # Spatial Sampling
    [x,y]=np.meshgrid(np.arange(-n/2, n/2), np.arange(-m/2, m/2));
    fx=(x/(ps*M));    #frequency space width [1/m]
    fy=(y/(ps*N));    #frequency space height [1/m]
    fx2fy2 = fx**2 + fy**2;
    
    
    ein_pad = ein;
    E0fft = np.fft.fftshift(np.fft.fft2(ein_pad));
    mask = 1;
    
    for z_ind in range(0,z.shape[0]):
        H  = np.exp(-1j*np.pi*lmda*z[z_ind]*fx2fy2); # Fast Transfer Function
        Eout_pad=np.fft.ifft2(np.fft.ifftshift(E0fft*H*mask));
        
        #f_metric[z_ind] = np.sum(np.abs(H)*np.sum(np.abs(E0fft)))
        f_metric[z_ind] = np.linalg.norm(np.real(H)*2*np.real(H)*E0fft,1)
        eout[:,:,z_ind]=Eout_pad #[1+(M-m)/2:(M+m)/2,1+(N-n)/2:(N+n)/2];
        Hout[:,:,z_ind]=H.copy()
    
    
    return(eout, Hout, f_metric)



def interactive_slider(image, title):
    from ipywidgets import widgets
    import matplotlib.pyplot as plt            # For making figures
    def slice_through_images(image):
        def slice_step(i_step):
            fig, axes = plt.subplots(figsize=(10, 5))
            axes.imshow(image[:,:,i_step], cmap='gray')
            plt.title(title)
            plt.colorbar
            plt.show()
        return slice_step  
    stepper = slice_through_images(image)
    widgets.interact(stepper, i_step=(0, image.shape[2]-1))
    
def imshowAnim(myimage, zs, niter, imsize):
    
    import matplotlib.pyplot as plt            # For making figures
    import numpy as np                         # Standard NumPy library 
    from matplotlib import animation, rc       # Used for inline animations
    from IPython.display import HTML           # Used for inline animations
    
    
    fig = plt.figure();
    fig=plt.figure(figsize=(imsize, imsize), dpi= 100, facecolor='w', edgecolor='k');
    a = (myimage[:,:,0]);
    im=plt.imshow(np.abs(myimage[:,:,0]), extent=[0, 1, 0, 1], vmin=np.min(myimage[:,:,0]), vmax=np.max(myimage[:,:,0]));
    ttl = plt.title(('Image # %s'%(1)));
    plt.axis('off');
    plt.close();

    def init():
        im.set_data(np.abs(myimage[:,:,0]));
        return [im]

    # animation function.  This is called sequentially
    def animate(i): 
        a = np.abs(myimage[:,:,i]);
        im.set_array(a);
        im.set_cmap('gray')
        
        ttl.set_text(('Defocus %s (um)'%(zs[i])));  # Change text 
        return [im, ttl]


    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=niter, interval=200);

    return(HTML(anim.to_html5_video()));