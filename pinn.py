import tensorflow as tf
from .layer import GradientLayer

class PINN:
    """
    Build a physics informed neural network (PINN) model for the transport equation.
    Attributes:
        network: keras network model with input (t, x) and output u(t, x).
        grads: gradient layer.
    """

    def __init__(self, network, c=1):
        """
        Args:
            network: keras network model with input (t, x) and output u(t, x).
            c: Default is 1.
        """

        self.network = network
        self.c = c
        self.grads = GradientLayer(self.network)

    def build(self):
        """
        Build a PINN model for the transport equation.
        Returns:
            PINN model with
                input: [ (t, x) relative to equation,
                         (t=0, x) relative to initial condition],
                output: [ u(t,x) relative to equation,
                          u(t=0, x) relative to initial condition]
        """

        # equation input: (t, x)
        tx_eqn = tf.keras.layers.Input(shape=(2,))
        # initial condition input: (t=0, x)
        tx_ini = tf.keras.layers.Input(shape=(2,))
        # boundary condition input: (t, x=-1) or (t, x=+1)

        # compute gradients
        u, du_dt, du_dx, d2u_dt2, d2u_dx2 = self.grads(tx_eqn)

        # compute f(u)
        #f = self.network(tx_bnd)*self.network(tx_bnd)*self.network(tx_bnd) - self.network(tx_bnd)   

        # equation output being zero
        u_eqn = du_dt + self.c * du_dx
        # initial condition output
        u_ini, du_dt_ini, _, _, _ = self.grads(tx_ini)
        # boundary condition output
        ###u_bnd = self.network(tx_bnd)  # dirichlet
        #_, _, u_bnd, _, _ = self.grads(tx_bnd)  # neumann


        # build the PINN model for the wave equation
        return tf.keras.models.Model(
            inputs=[tx_eqn, tx_ini],
            outputs=[u_eqn, u_ini])
