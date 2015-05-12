"""
Implementation of ADASECANT training algorithm

'ADASECANT: Robust Adaptive Secant Method for Stochastic Gradient'
Caglar Gulcehre, Yoshua Bengio
http://arxiv.org/abs/1412.7419

Code adapted from Caglar Gulcehre:
https://github.com/caglar/adasecant_wshp_paper

.. note::
    From the author
    <http://www.reddit.com/r/MachineLearning/comments/2w6bfl/ask_ml_has_anyone_tried_to_reproduce_adasecant/>:
    This implementation on github is slightly different from the one mentioned in the paper,
    there are some minor differences (outlier detection mechanism is slightly different). But in principle those
    shouldn't change the results too much... The important point to care about in the implementation of this algorithm,
    is the initialization of the accumulators. You can check the code to see how I initialized it.
    Please also check my comments as well. Note that it is important to keep the rms statistics of gamma as well
    (it should not be negative).

    Please note that this was just a workshop paper and the algorithm is still work in progress.
    We plan to publish another paper related to that with some collaborators.

"""

__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Caglar Gulcehre", "Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
from collections import OrderedDict
# third party libraries
import theano.tensor as T
import numpy
# internal references
from opendeep import sharedX
from opendeep.optimization.optimizer import Optimizer

log = logging.getLogger(__name__)


# All AdaSecant needs to do is implement the get_updates() method for stochastic gradient descent
class AdaSecant(Optimizer):
    """
    Taken from https://github.com/caglar/adasecant_wshp_paper:
    Adasecant:
    Based on the paper:
    Gulcehre, Caglar, and Yoshua Bengio.
    "ADASECANT: Robust Adaptive Secant Method for Stochastic Gradient."
    arXiv preprint arXiv:1412.7419 (2014).
    There are some small changes in this code.
    """
    def __init__(self, model, dataset,
                 n_epoch=10, batch_size=100, minimum_batch_size=1,
                 save_frequency=None, early_stop_threshold=None, early_stop_length=None,
                 learning_rate=1e-6, lr_decay=None, lr_factor=None,
                 decay=0.95, gamma_clip=1.8, damping=1e-7, grad_clip=None, start_var_reduction=0,
                 delta_clip=None, use_adagrad=False, skip_nan_inf=False,
                 upper_bound_tau=1e8, lower_bound_tau=1.5, use_corrected_grad=True):
        """
        Initialize AdaSecant.

        Parameters
        ----------
        model : Model
            The Model to train.
        dataset : Dataset
            The Dataset to use when training the Model.
        n_epoch : int
            how many training iterations over the dataset to go.
        batch_size : int
            How many examples from the training dataset to use in parallel.
        minimum_batch_size : int
            The minimum number of examples required at a time (for things like time series, this would be > 1).
        save_frequency : int
            How many epochs to train between each new save of the Model's parameters.
        early_stop_threshold : float
            The factor by how much the best validation training score needs to improve to determine early stopping.
        early_stop_length : int
            The patience or number of epochs to wait after the early_stop_threshold has been reached before stopping.
        learning_rate : float
            The multiplicative amount to adjust parameters based on their gradient values.
        lr_decay : str
            The type of decay function to use for changing the learning rate over epochs. See
            `opendeep.utils.decay` for options.
        lr_factor : float
            The amount to use for the decay function when changing the learning rate over epochs. See
            `opendeep.utils.decay` for its effect for given decay functions.
        decay : float, optional
            Decay rate :math:`\\rho` in Algorithm 1 of the aforementioned
            paper. Decay 0.95 seems to work fine for several tasks.
        gamma_clip : float, optional
            The clipping threshold for the gamma. In general 1.8 seems to
            work fine for several tasks.
        start_var_reduction: float, optional,
            How many updates later should the variance reduction start from?
        delta_clip: float, optional,
            The threshold to clip the deltas after.
        grad_clip: float, optional,
            Apply gradient clipping for RNNs (not necessary for feedforward networks). But this is
            a constraint on the norm of the gradient per layer.
            Based on:
            Pascanu, Razvan, Tomas Mikolov, and Yoshua Bengio. "On the difficulty of training
            recurrent neural networks." arXiv preprint arXiv:1211.5063 (2012).
        use_adagrad: bool, optional
            Either to use clipped adagrad or not.
        use_corrected_grad: bool, optional
            Either to use correction for gradients (referred as variance
            reduction in the workshop paper).
        """
        # get everything together with the Optimizer class
        initial_parameters = locals().copy()
        initial_parameters.pop('self')
        super(AdaSecant, self).__init__(**initial_parameters)

        assert decay >= 0., "Decay needs to be >=0."
        assert decay < 1., "Decay needs to be <1."
        self.decay = sharedX(decay, "decay")

        self.damping = damping
        self.skip_nan_inf = skip_nan_inf

        if grad_clip:
            assert grad_clip > 0.
            assert grad_clip <= 1., "Norm of the gradients per layer can not be larger than 1."
        self.grad_clip = grad_clip

        self.use_adagrad = use_adagrad
        self.use_corrected_grad = use_corrected_grad
        self.gamma_clip = gamma_clip
        self.start_var_reduction = start_var_reduction
        self.delta_clip = delta_clip

        # We have to bound the tau to prevent it to
        # grow to an arbitrarily large number, oftenwise
        # that causes numerical instabilities for very deep
        # networks. Note that once tau become very large, it will keep,
        # increasing indefinitely.
        self.lower_bound_tau = lower_bound_tau
        self.upper_bound_tau = upper_bound_tau

    def get_updates(self, gradients):
        """
        Compute AdaSecant updates (see the paper for details).

        Parameters
        ----------
        gradients : dict
            A dictionary mapping from the model's parameters to their gradients.

        Returns
        -------
        OrderdDict
            A dictionary mapping from the old model parameters
            to their new values after a single iteration of the learning rule.
        """
        updates = OrderedDict({})
        eps = self.damping
        step = sharedX(0., name="step")

        if self.skip_nan_inf:
            #If norm of the gradients of a parameter is inf or nan don't update that parameter
            #That might be useful for RNNs.
            gradients = OrderedDict({p: T.switch(T.or_(T.isinf(gradients[p]),
                T.isnan(gradients[p])), 0, gradients[p]) for
                p in gradients.keys()})

        #Block-normalize gradients:
        gradients = OrderedDict({p: gradients[p] / (gradients[p].norm(2) + eps) for p in gradients.keys()})
        nparams = len(gradients.keys())

        #Apply the gradient clipping, this is only necessary for RNNs and sometimes for very deep
        #networks
        if self.grad_clip:
            gnorm = sum([g.norm(2) for g in gradients.values()])

            gradients = OrderedDict({p: T.switch(gnorm/nparams > self.grad_clip,
                                 g * self.grad_clip * nparams / gnorm , g)\
                                 for p, g in gradients.iteritems()})

        for param in gradients.keys():
            gradients[param].name = "grad_%s" % param.name
            mean_grad = sharedX(param.get_value() * 0. + eps, name="mean_grad_%s" % param.name)
            # mean_corrected_grad = sharedX(param.get_value() * 0 + eps, name="mean_corrected_grad_%s" % param.name)
            slow_constant = 2.1

            if self.use_adagrad:
                # sum_square_grad := \sum_i g_i^2
                sum_square_grad = sharedX(param.get_value(borrow=True) * 0., name="sum_square_grad_%s" % param.name)

            """
            Initialization of accumulators
            """
            taus_x_t = sharedX((numpy.ones_like(param.get_value()) + eps) * slow_constant,
                               name="taus_x_t_" + param.name)
            self.taus_x_t = taus_x_t

            #Variance reduction parameters
            #Numerator of the gamma:
            gamma_nume_sqr = sharedX(numpy.zeros_like(param.get_value()) + eps,
                                     name="gamma_nume_sqr_" + param.name)

            #Denominator of the gamma:
            gamma_deno_sqr = sharedX(numpy.zeros_like(param.get_value()) + eps,
                                     name="gamma_deno_sqr_" + param.name)

            #For the covariance parameter := E[\gamma \alpha]_{t-1}
            cov_num_t = sharedX(numpy.zeros_like(param.get_value()) + eps,
                                name="cov_num_t_" + param.name)

            # mean_squared_grad := E[g^2]_{t-1}
            mean_square_grad = sharedX(numpy.zeros_like(param.get_value()) + eps,
                                       name="msg_" + param.name)

            # mean_square_dx := E[(\Delta x)^2]_{t-1}
            mean_square_dx = sharedX(param.get_value() * 0., name="msd_" + param.name)
            if self.use_corrected_grad:
                old_grad = sharedX(param.get_value() * 0. + eps)

            #The uncorrected gradient of previous of the previous update:
            old_plain_grad = sharedX(param.get_value() * 0. + eps)
            mean_curvature = sharedX(param.get_value() * 0. + eps)
            mean_curvature_sqr = sharedX(param.get_value() * 0. + eps)

            # Initialize the E[\Delta]_{t-1}
            mean_dx = sharedX(param.get_value() * 0.)

            # Block-wise normalize the gradient:
            norm_grad = gradients[param]

            #For the first time-step, assume that delta_x_t := norm_grad
            cond = T.eq(step, 0)
            msdx = cond * norm_grad**2 + (1 - cond) * mean_square_dx
            mdx  = cond * norm_grad + (1 - cond) * mean_dx

            """
            Compute the new updated values.
            """
            # E[g_i^2]_t
            new_mean_squared_grad = (mean_square_grad * self.decay + T.sqr(norm_grad) * (1 - self.decay))
            new_mean_squared_grad.name = "msg_" + param.name
            # E[g_i]_t
            new_mean_grad = (mean_grad * self.decay + norm_grad * (1 - self.decay))
            new_mean_grad.name = "nmg_" + param.name

            mg = new_mean_grad
            mgsq = new_mean_squared_grad

            # Keep the rms for numerator and denominator of gamma.
            new_gamma_nume_sqr = (
                gamma_nume_sqr * (1 - 1 / taus_x_t) + T.sqr((norm_grad - old_grad) * (old_grad - mg)) / taus_x_t
            )
            new_gamma_nume_sqr.name = "ngammasqr_num_" + param.name

            new_gamma_deno_sqr = (
                gamma_deno_sqr * (1 - 1 / taus_x_t) + T.sqr((mg - norm_grad) * (old_grad - mg)) / taus_x_t
            )
            new_gamma_deno_sqr.name = "ngammasqr_den_" + param.name

            gamma = T.sqrt(gamma_nume_sqr) / T.sqrt(gamma_deno_sqr + eps)
            gamma.name = "gamma_" + param.name

            if self.gamma_clip:
                gamma = T.minimum(gamma, self.gamma_clip)

            momentum_step = gamma * mg
            corrected_grad_cand = (norm_grad + momentum_step) / (1 + gamma)

            #For starting the variance reduction.
            if self.start_var_reduction > -1:
                cond = T.le(self.start_var_reduction, step)
                corrected_grad = cond * corrected_grad_cand + (1 - cond) * norm_grad
            else:
                corrected_grad = norm_grad

            new_sum_squared_grad = None
            if self.use_adagrad:
                g = corrected_grad
                # Accumulate gradient
                new_sum_squared_grad = (sum_square_grad + T.sqr(g))

                rms_g_t = T.sqrt(new_sum_squared_grad)
                rms_g_t = T.maximum(rms_g_t, 1.0)

            # Use the gradients from the previous update
            # to compute the \nabla f(x_t) - \nabla f(x_{t-1})
            cur_curvature = norm_grad - old_plain_grad
            cur_curvature_sqr = T.sqr(cur_curvature)

            new_curvature_ave = (mean_curvature * (1 - 1 / taus_x_t) + (cur_curvature / taus_x_t))
            new_curvature_ave.name = "ncurve_ave_" + param.name

            #Average average curvature
            nc_ave = new_curvature_ave

            new_curvature_sqr_ave = (mean_curvature_sqr * (1 - 1 / taus_x_t) + (cur_curvature_sqr / taus_x_t))
            new_curvature_sqr_ave.name = "ncurve_sqr_ave_" + param.name

            #Unbiased average squared curvature
            nc_sq_ave = new_curvature_sqr_ave

            epsilon = self.lr_scalers.get(param, 1.) * self.learning_rate
            scaled_lr = self.lr_scalers.get(param, 1.) * sharedX(1.0)
            rms_dx_tm1 = T.sqrt(msdx + epsilon)

            rms_curve_t = T.sqrt(new_curvature_sqr_ave + epsilon)

            #This is where the update step is being defined
            delta_x_t = -scaled_lr * (rms_dx_tm1 / rms_curve_t - cov_num_t / (new_curvature_sqr_ave + epsilon))
            delta_x_t.name = "delta_x_t_" + param.name

            # This part seems to be necessary for only RNNs
            # For feedforward networks this does not seem to be important.
            if self.delta_clip:
                log.info("Clipping will be applied on the adaptive step size.")
                delta_x_t = delta_x_t.clip(-self.delta_clip, self.delta_clip)
                if self.use_adagrad:
                    delta_x_t = delta_x_t * corrected_grad / rms_g_t
                else:
                    log.info("Clipped adagrad is disabled.")
                    delta_x_t = delta_x_t * corrected_grad
            else:
                log.info("Clipping will not be applied on the adaptive step size.")
                if self.use_adagrad:
                    delta_x_t = delta_x_t * corrected_grad / rms_g_t
                else:
                    log.info("Clipped adagrad will not be used.")
                    delta_x_t = delta_x_t * corrected_grad

            new_taus_t = (1 - T.sqr(mdx) / (msdx + eps)) * taus_x_t + sharedX(1 + eps, "stabilized")

            #To compute the E[\Delta^2]_t
            new_mean_square_dx = (msdx * (1 - 1 / taus_x_t) + (T.sqr(delta_x_t) / taus_x_t))

            #To compute the E[\Delta]_t
            new_mean_dx = (mean_dx * (1 - 1 / taus_x_t) + (delta_x_t / (taus_x_t)))

            #Perform the outlier detection:
            #This outlier detection is slightly different:
            new_taus_t = T.switch(T.or_(abs(norm_grad - mg) > (2 * T.sqrt(mgsq - mg**2)),
                                        abs(cur_curvature - nc_ave) > (2 * T.sqrt(nc_sq_ave - nc_ave**2))),
                                        sharedX(2.2), new_taus_t)

            #Apply the bound constraints on tau:
            new_taus_t = T.maximum(self.lower_bound_tau, new_taus_t)
            new_taus_t = T.minimum(self.upper_bound_tau, new_taus_t)

            new_cov_num_t = (cov_num_t * (1 - 1 / taus_x_t) + (delta_x_t * cur_curvature) * (1 / taus_x_t))

            update_step = delta_x_t

            # Apply updates
            updates[mean_square_grad] = new_mean_squared_grad
            updates[mean_square_dx] = new_mean_square_dx
            updates[mean_dx] = new_mean_dx
            updates[gamma_nume_sqr] = new_gamma_nume_sqr
            updates[gamma_deno_sqr] = new_gamma_deno_sqr
            updates[taus_x_t] = new_taus_t
            updates[cov_num_t] = new_cov_num_t
            updates[mean_grad] = new_mean_grad
            updates[old_plain_grad] = norm_grad
            updates[mean_curvature] = new_curvature_ave
            updates[mean_curvature_sqr] = new_curvature_sqr_ave
            updates[param] = param + update_step
            updates[step] = step + 1

            if self.use_adagrad:
                updates[sum_square_grad] = new_sum_squared_grad

            if self.use_corrected_grad:
                updates[old_grad] = corrected_grad

        return updates