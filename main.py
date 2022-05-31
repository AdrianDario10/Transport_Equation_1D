import lib.tf_silent
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from lib.pinn import PINN
from lib.network import Network
from lib.optimizer import L_BFGS_B
from numpy import linalg as LA

def u0(tx, c=1, k=2, sd=0.5):
    """
    Initial wave form.
    Args:
        tx: variables (t, x) as tf.Tensor.
        c: wave velocity.
        k: wave number.
        sd: standard deviation.
    Returns:
        u(t, x) as tf.Tensor.
    """

    t = tx[..., 0, None]
    x = tx[..., 1, None]
    #z = k*x - (c*k)*t


    return   1/(1+x**2) #tf.sin(z) * tf.exp(-(0.5*z/sd)**2)  #x*x*x-x   #tf.cos(2*x+t)  

#def du0_dt(tx):
#    """
#    First derivative of t for the initial wave form.
#    Args:
#        tx: variables (t, x) as tf.Tensor.
#    Returns:
#        du(t, x)/dt as tf.Tensor.
#    """
#
#    with tf.GradientTape() as g:
#        g.watch(tx)
#        u = u0(tx)
#    du_dt = g.batch_jacobian(u, tx)[..., 0]
#    return du_dt

if __name__ == '__main__':
    """
    Test the physics informed neural network (PINN) model for the wave equation.
    """
    import numpy as np


    def primes_from_2_to(n):
      sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool)
      for i in range(1, int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
      return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


    def van_der_corput(n_sample, base=2):
      sequence = []
      for i in range(n_sample):
        n_th_number, denom = 0., 1.
        while i > 0:
            i, remainder = divmod(i, base)
            denom *= base
            n_th_number += remainder / denom
        sequence.append(n_th_number)

      return sequence


    def halton(dim, n_sample):
      big_number = 10
      while 'Not enought primes':
        base = primes_from_2_to(big_number)[:dim]
        if len(base) == dim:
            break
        big_number += 1000

    # Generate a sample using a Van der Corput sequence per dimension.
      sample = [van_der_corput(n_sample + 1, dim) for dim in base]
      sample = np.stack(sample, axis=-1)[1:]

      return sample

    # number of training samples
    num_train_samples = 10000
    # number of test samples
    num_test_samples = 1000

    # build a core network model
    network = Network.build()
    network.summary()
    # build a PINN model
    pinn = PINN(network).build()

    # Time and space domain
    t=3
    x_f=3
    x_ini=-3
    num = np.sqrt(num_train_samples)
    num = int(np.round(num,0))

    epsilon=1e-4
    x_=np.linspace(x_ini +epsilon,x_f-epsilon,num)
    t_=np.linspace(0+epsilon,t-epsilon,num)

    T, X = np.meshgrid(t_,x_)

    #tx_eqn=np.random.rand(num**2, 2)
    #tx_eqn[...,0]= T.reshape((num**2,))
    #tx_eqn[...,1]= X.reshape((num**2,))

    # create training input
    tx_eqn = np.random.rand(num_train_samples, 2)#halton(2, num_train_samples)#np.random.rand(num_train_samples, 2)
    tx_eqn[..., 0] = t*tx_eqn[..., 0]                # t =  0 ~ +4
    tx_eqn[..., 1] = (x_f-x_ini)*tx_eqn[..., 1] + x_ini            # x = -1 ~ +1
    tx_ini = np.random.rand(num_train_samples, 2)
    tx_ini[..., 0] = 0                               # t = 0
    tx_ini[..., 1] = (x_f-x_ini)*tx_ini[..., 1] + x_ini            # x = -1 ~ +1
    # create training output
    u_zero = np.zeros((num_train_samples, 1))
    u_ini = u0(tf.constant(tx_ini)).numpy()
    ##du_dt_ini = du0_dt(tf.constant(tx_ini)).numpy()

    # train the model using L-BFGS-B algorithm
    x_train = [tx_eqn, tx_ini]
    y_train = [u_zero, u_ini]
    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
    lbfgs.fit()

    # predict u(t,x) distribution
    t_flat = np.linspace(0, t-1, num_test_samples)
    x_flat = np.linspace(x_ini+1, x_f-1, num_test_samples)
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    u = network.predict(tx, batch_size=num_test_samples)
    u = u.reshape(t.shape)
    

    
    U = 1/(1+(x-t)**2)* np.exp(-t/2)
    E = (U-u)
    print(np.max(np.max(np.abs(E))))
    



    
    # plot u(t,x) distribution as a color-map

    fig= plt.figure(figsize=(15,10))
    vmin, vmax = 0, 1
    plt.pcolormesh(t, x, u, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
    font1 = {'family':'serif','size':20}
    font2 = {'family':'serif','size':15}

    plt.title("u(x,t)", fontdict = font1)
    plt.xlabel("t", fontdict = font1)
    plt.ylabel("x", fontdict = font1)
    plt.tick_params(axis='both', which='major', labelsize=15)

    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(x,t)', fontdict = font1)
    cbar.mappable.set_clim(vmin, vmax)
    cbar.ax.tick_params(labelsize=15)
    plt.show()

    print('Model in 3D:\n')
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator

    # ERROR

    U = 1/(1+(x-t)**2)
    E = (U-u)
    print('Error surface:\n')
    #MSE=np.sum(np.sum(E))/num_test_samples
    print('\n')
    print(np.max(np.max(np.abs(E))))
    print('\n')
    
    
    fig= plt.figure(figsize=(15,10))
    vmin, vmax = np.min(np.min(E)), np.max(np.max(E))
    plt.pcolormesh(t, x, E, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
    font1 = {'family':'serif','size':20}
    font2 = {'family':'serif','size':15}

    plt.title("Error", fontdict = font1)
    plt.xlabel("t", fontdict = font1)
    plt.ylabel("x", fontdict = font1)
    plt.tick_params(axis='both', which='major', labelsize=15)

    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('Error', fontdict = font1)
    cbar.mappable.set_clim(vmin, vmax)
    cbar.ax.tick_params(labelsize=15)
    plt.show()

    # Comparison

    fig,(ax1, ax2, ax3)  = plt.subplots(1,3,figsize=(15,6))
    x_flat_ = np.linspace(x_ini+1, x_f-1, 10)

   
    U_1 = 1/(1+(x_flat_)**2)
    tx = np.stack([np.full(t_flat.shape, 0), x_flat], axis=-1)
    u_ = network.predict(tx, batch_size=num_test_samples)
    ax1.plot(x_flat, u_)
    ax1.plot(x_flat_, U_1,'r*')
    font1 = {'family':'serif','size':20}
    font2 = {'family':'serif','size':15}

    ax1.set_title('t={}'.format(0), fontdict = font1)
    ax1.set_xlabel('x', fontdict = font1)
    ax1.set_ylabel('u(t,x)', fontdict = font1)
    ax1.tick_params(labelsize=15)
    #plt.show()
    print('\n')

    
    U_1 = 1/(1+(x_flat_-1)**2) 
    tx = np.stack([np.full(t_flat.shape, 1), x_flat], axis=-1)
    u_ = network.predict(tx, batch_size=num_test_samples)
    ax2.plot(x_flat, u_)
    ax2.plot(x_flat_, U_1,'r*')
    ax2.set_title('t={}'.format(1), fontdict = font1)
    ax2.set_xlabel('x', fontdict = font1)
    ax2.set_ylabel('u(t,x)', fontdict = font1)
    ax2.tick_params(labelsize=15)
    #plt.show()
    print('\n')

    
    U_1 = 1/(1+(x_flat_-2)**2) 
    tx = np.stack([np.full(t_flat.shape, 2), x_flat], axis=-1)
    u_ = network.predict(tx, batch_size=num_test_samples)
    ax3.plot(x_flat, u_,label='Computed solution')
    ax3.plot(x_flat_, U_1,'r*', label='Exact solution')
    ax3.set_title('t={}'.format(2), fontdict = font1)
    ax3.set_xlabel('x', fontdict = font1)
    ax3.set_ylabel('u(t,x)', fontdict = font1)
    ax3.legend(loc='best', fontsize = 'xx-large')
    ax3.tick_params(labelsize=15)
    plt.tight_layout()
    plt.show()
    
