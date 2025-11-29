import warnings
warnings.filterwarnings("ignore")

# basics
import math
import numpy as np


##############################################
"""
    Parameters:
    -----------
    signal : ndarray
        Input multivariate signal to be decomposed (channels x samples)
    num_modes : int
        Number of modes to be recovered.
    alpha : float
        Bandwidth constraint parameter.
    tolerance : float
        Stopping criterion for the dual ascent
    init : int
        Initialization method for center frequencies:
            - 0: All omegas start at 0.
            - 1: All omegas are initialized lineary distributed.
            - 2: All omegas are initialized exponentially distributed.
    tau : float
        Time-step of the dual ascent (use 0 for noise-slack).
    lambda_param : float
        Lagrange multiplier weight.
    DC : bool
        True if the first mode is to be held at DC (0-freq).

    Returns:
    --------
    modes : ndarray
        The collection of decomposed modes (K x T).
    modes_hat : ndarray
        Spectra of the modes (K x F).
    omega : ndarray
        Estimated mode center-frequencies (iter x K).
"""

def VMD(signal, alpha, tau, K, DC, init, tol, lambda_param):

    # Period and sampling frequency of input signal
    save_T=len(signal)
    fs=1/float(save_T)

    # extend the signal by mirroring
    T=save_T
    f_mirror=np.zeros(2*T)
    f_mirror[0:T//2]=signal[T//2-1::-1]
    f_mirror[T//2:3*T//2]= signal
    f_mirror[3*T//2:2*T]=signal[-1:-T//2-1:-1]
    f=f_mirror
    print('-------')

    # Time Domain 0 to T (of mirrored signal)
    T=float(len(f))
    t=np.linspace(1/float(T),1,int(T),endpoint=True)

    # Spectral Domain discretization
    freqs=t-0.5-1/T

    # Maximum number of iterations (if not converged yet, then it won't anyway)
    N=500

    # For future generalizations: individual alpha for each mode
    Alpha=alpha*np.ones(K,dtype=complex)

    # Construct and center f_hat
    f_hat=np.fft.fftshift(np.fft.fft(f))
    f_hat_plus=f_hat
    f_hat_plus[0:int(int(T)/2)]=0

    # matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus=np.zeros((N,len(freqs),K),dtype=complex)

    # Initialization of omega_k
    omega_plus=np.zeros((N,K),dtype=complex)

    if (init==1):
        for i in range(1,K+1):
            omega_plus[0,i-1]=(0.5/K)*(i-1)
    elif (init==2):
        omega_plus[0,:]=np.sort(math.exp(math.log(fs))+(math.log(0.5)-math.log(fs))*np.random.rand(1,K))
    else:
        omega_plus[0,:]=0

    if (DC):
        omega_plus[0,0]=0
    
    # start with empty dual variables
    lamda_hat=lambda_param*np.ones((N,len(freqs)),dtype=complex)

    # other inits
    uDiff=tol+2.2204e-16 #update step
    n=1 #loop counter
    sum_uk=0 #accumulator

    T=int(T)


    # ----------- Main loop for iterative updates

    while uDiff > tol and n<N:

        # update first mode accumulator
        k=1; sum_uk = u_hat_plus[n-1,:,K-1]+sum_uk-u_hat_plus[n-1,:,0]

        #update spectrum of first mode through Wiener filter of residuals
        u_hat_plus[n,:,k-1]=(f_hat_plus-sum_uk-lamda_hat[n-1,:]/2)/(1+Alpha[k-1]*np.square(freqs-omega_plus[n-1,k-1]))
        
        #update first omega if not held at 0
        if DC==False:
            omega_plus[n,k-1]=np.dot(freqs[T//2:T],np.square(np.abs(u_hat_plus[n,T//2:T,k-1])).T)/np.sum(np.square(np.abs(u_hat_plus[n,T//2:T,k-1])))

        for k in range(2,K+1):

            #accumulator
            sum_uk=u_hat_plus[n,:,k-2]+sum_uk-u_hat_plus[n-1,:,k-1]

            #mode spectrum
            u_hat_plus[n,:,k-1]=(f_hat_plus-sum_uk-lamda_hat[n-1,:]/2)/(1+Alpha[k-1]*np.square(freqs-omega_plus[n-1,k-1]))
            
            #center frequencies
            omega_plus[n,k-1]=np.dot(freqs[T//2:T],np.square(np.abs(u_hat_plus[n,T//2:T,k-1])).T)/np.sum(np.square(np.abs(u_hat_plus[n,T//2:T:,k-1])))

        #Dual ascent
        lamda_hat[n,:]=lamda_hat[n-1,:]+tau*(np.sum(u_hat_plus[n,:,:],axis=1)-f_hat_plus)

        #loop counter
        n=n+1

        #converged yet?
        uDiff=2.2204e-16

        for i in range(1,K+1):
            uDiff=uDiff+1/float(T)*np.dot(u_hat_plus[n-1,:,i-1]-u_hat_plus[n-2,:,i-1],(np.conj(u_hat_plus[n-1,:,i-1]-u_hat_plus[n-2,:,i-1])).conj().T)
  
        uDiff=np.abs(uDiff)

    # ------ Postprocessing and cleanup

    # discard empty space if converged early

    N=np.minimum(N,n)
    omega = omega_plus[0:N,:]

    # Signal reconstruction
    u_hat = np.zeros((T,K),dtype=complex)
    u_hat[T//2:T,:]= np.squeeze(u_hat_plus[N-1,T//2:T,:])
    u_hat[T//2:0:-1,:]=np.squeeze(np.conj(u_hat_plus[N-1,T//2:T,:]))
    u_hat[0,:]=np.conj(u_hat[-1,:])
    u=np.zeros((K,len(t)),dtype=complex)

    for k in range(1,K+1):
        u[k-1,:]= np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:,k-1])))

    # remove mirror part 
    u=u[:,T//4:3*T//4]

    #recompute spectrum
    u_hat = np.zeros((T//2,K),dtype=complex)

    for k in range(1,K+1):
        u_hat[:,k-1]=np.fft.fftshift(np.fft.fft(u[k-1,:])).conj().T

    return u,u_hat,omega

