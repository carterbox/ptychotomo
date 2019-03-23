import solver
import dxchange
import objects
import numpy as np
import cupy as cp
import signal
import sys
import os
import sys


if __name__ == "__main__":
    igpu = np.int(sys.argv[1])
    cp.cuda.Device(igpu).use()  # gpu id to use
    print("gpu id:",igpu)

    
    # Model parameters
    voxelsize = 1e-6/2  # object voxel size
    energy = 8.8  # xray energy
    maxinta = [3,0.3,0.03,0.003]  # maximal probe intensity
    prbsize = 16 # probe size
    prbshift = 12  # probe shift (probe overlap = (1-prbshift)/prbsize)
    det = [128, 128] # detector size
    n = 512 #object size
    ntheta = 256*3//2
    noise = True  # apply discrete Poisson noise
    
    # Reconstrucion parameters
    modela = ['poisson']  # minimization funcitonal (poisson,gaussian)
    alphaa = [8e-9,1e-8,2e-8,4e-8] # tv regularization penalty coefficient
    maxint = maxinta[igpu]    
    alphaa = [alphaa[igpu]]
    piter = 4  # ptychography iterations
    titer = 4  # tomography iterations
    NITER = 300  # ADMM iterations

    ptheta = 32 # NEW: number of angular partitions for simultaneous processing in ptychography
    
    name = 'noise'+str(noise)+'maxint' + \
        str(maxint)+'prbshift'+str(prbshift)+'ntheta'+str(ntheta)

    scan = cp.array(np.load('/mxn/home/viknik/ptychotomo/gendata/coordinates'+name+'.npy'))
    data = np.load('/mxn/home/viknik/ptychotomo/gendata/data'+name+'.npy')
    prb = cp.array(np.load('/mxn/home/viknik/ptychotomo/gendata/prb'+name+'.npy'))
    theta = cp.array(np.load('/mxn/home/viknik/ptychotomo/gendata/theta'+name+'.npy'))

    objshape=[2*prbshift+prbsize,n,n]
    tomoshape=[ntheta,2*prbshift+prbsize,n]

    # Class gpu solver 
    slv = solver.Solver(prb, scan, theta, det, voxelsize, energy, tomoshape, ptheta)
    # Free gpu memory after SIGINT, SIGSTSTP
    def signal_handler(sig, frame):
        slv = []
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)



    for imodel in range(len(modela)):
        model = modela[imodel]
        for ialpha in range(len(alphaa)):
            alpha = alphaa[ialpha]

            # Initial guess
            h = cp.zeros(tomoshape, dtype='complex64', order='C')+1
            psi = cp.zeros(tomoshape, dtype='complex64', order='C')+1
            e = cp.zeros([3, *objshape], dtype='complex64', order='C')
            phi = cp.zeros([3, *objshape], dtype='complex64', order='C')
            lamd = cp.zeros(tomoshape, dtype='complex64', order='C')
            mu = cp.zeros([3, *objshape], dtype='complex64', order='C')
            u = cp.zeros(objshape, dtype='complex64', order='C')

            # ADMM
            u, psi, lagr = slv.admm(data, h, e, psi, phi, lamd,
                                    mu, u, alpha, piter, titer, NITER, model)
             # Save result
            name = 'reg'+str(alpha)+'noise'+str(noise)+'maxint' + \
                str(maxint)+'prbshift'+str(prbshift)+'ntheta'+str(ntheta)+str(model)+str(piter)+str(titer)+str(NITER)

            dxchange.write_tiff(u.imag.get(),  'beta/beta'+name)
            dxchange.write_tiff(u.real.get(),  'delta/delta'+name)
            if not os.path.exists('lagr'):
                os.makedirs('lagr')
            np.save('lagr/lagr'+name,lagr.get())
            if(model=='gaussian'):
                exit()
