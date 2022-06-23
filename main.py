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
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# Initial condition function
def u0(tx):
    """
    Initial wave form.
    Args:
        tx: variables (t, x) as tf.Tensor.
    Returns:
        u(t, x) as tf.Tensor.
    """

    t = tx[..., 0, None]
    x = tx[..., 1, None]

    return   1/(1+x**2) 

# Halton distribution
def primes_from_2_to(n):
    """Prime number from 2 to n.
    From `StackOverflow <https://stackoverflow.com/questions/2068372>`_.
    :param int n: sup bound with ``n >= 6``.
    :return: primes in 2 <= p < n.
    :rtype: list
    """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def van_der_corput(n_sample, base=2):
    """Van der Corput sequence.
    :param int n_sample: number of element of the sequence.
    :param int base: base of the sequence.
    :return: sequence of Van der Corput.
    :rtype: list (n_samples,)
    """
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
    """Halton sequence.
    :param int dim: dimension
    :param int n_sample: number of samples.
    :return: sequence of Halton.
    :rtype: array_like (n_samples, n_features)
    """
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


if __name__ == '__main__':
    """
    Test the physics informed neural network (PINN) model for the wave equation.
    """
    
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
    t=2
    x_f=2
    x_ini=-2
    
    ''' Example of parametric distribution over the domain:
    
    num = np.sqrt(num_train_samples)
    num = int(np.round(num,0))

    epsilon=1e-3
    x=np.linspace(-2+epsilon,2-epsilon,num)
    t=np.linspace(0+epsilon,2-epsilon,num)

    T, X = np.meshgrid(t,x)

    tx_eqn=np.random.rand(num**2, 2)
    tx_eqn[...,0]=T.reshape((num**2,))
    tx_eqn[...,1]=X.reshape((num**2,))
    '''
    
    ''' Example of parametric distribution over the domain:
    
    tx_eqn = halton(2, num_train_samples)
    tx_eqn[..., 0] = t*tx_eqn[..., 0]            
    tx_eqn[..., 1] = (x_f-x_ini)*tx_eqn[..., 1] + x_ini          
    '''
    
    # create training input
    tx_eqn = np.random.rand(num_train_samples, 2)
    tx_eqn[..., 0] = t*tx_eqn[..., 0]                # t =  0 ~ +2
    tx_eqn[..., 1] = (x_f-x_ini)*tx_eqn[..., 1] + x_ini            # x = -2 ~ +2
    tx_ini = np.random.rand(num_train_samples, 2)
    tx_ini[..., 0] = 0                               # t = 0
    tx_ini[..., 1] = (x_f-x_ini)*tx_ini[..., 1] + x_ini            # x = -1 ~ +1
    # create training output
    u_zero = np.zeros((num_train_samples, 1))
    u_ini = u0(tf.constant(tx_ini)).numpy()


    # train the model using L-BFGS-B algorithm
    x_train = [tx_eqn, tx_ini]
    y_train = [u_zero, u_ini]
    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
    lbfgs.fit()

    # prediction of u(t,x) distribution
    t_flat = np.linspace(0, t, num_test_samples)
    x_flat = np.linspace(x_ini, x_f, num_test_samples)
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    u = network.predict(tx, batch_size=num_test_samples)
    u = u.reshape(t.shape)
    
    
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

    # Exact solution U and Error E
    U = 1/(1+(x-t)**2)
    E = (U-u)
    
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

    # Comparison at time 0, 1 and 2

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

    
    U_1 = 1/(1+(x_flat_-1)**2) 
    tx = np.stack([np.full(t_flat.shape, 1), x_flat], axis=-1)
    u_ = network.predict(tx, batch_size=num_test_samples)
    ax2.plot(x_flat, u_)
    ax2.plot(x_flat_, U_1,'r*')
    ax2.set_title('t={}'.format(1), fontdict = font1)
    ax2.set_xlabel('x', fontdict = font1)
    ax2.set_ylabel('u(t,x)', fontdict = font1)
    ax2.tick_params(labelsize=15)

    
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
    
