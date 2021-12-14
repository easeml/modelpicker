import numpy as np
"""This code runs stream based model picker, but suitable for pool setting as well."""

def model_picker(predictions, labelset, budget):
    """
    :param predictions:
    :param labelset:
    :param budget:
    :return:
    """
    # Set params
    num_models = np.size(predictions, 2)
    num_instances = np.size(predictions, 1)

    # Initializations
    eta_0 = np.sqrt(np.log(num_models)/2) # initialize the hyperparameter
    bias_scale = 1 # this linear scaling parameter is added to boost the coin tossing bias in case needed

    # Edit the input data accordingly with the shuffled indices
    shuffled_indices =
    predictions = predictions[shuffled_indices, :]


    # Initialize
    loss_t = np.zeros(data._num_models) # loss per models
    z_t_log = np.zeros(data._num_instances, dtype=int) # binary query decision
    z_t_budget = np.zeros(data._num_instances, dtype=int) # binary query decision
    posterior_t_log = np.zeros((data._num_instances, data._num_models)) # posterior log
    mp_oracle = np.zeros(data._num_instances)
    hidden_loss_log = np.zeros(data._num_instances, dtype=int)
    It_log = np.zeros(data._num_instances, dtype=int)
    posterior_t = np.ones(data._num_models)/data._num_models

    # For each streaming data instance
    for t in np.arange(1, data._num_instances+1, 1):

        # Edit eta
        eta = eta_0 / np.sqrt(t)


        posterior_t = np.exp(-eta * (loss_t-np.min(loss_t)))
        # Note that above equation is equivalent to np.exp(-eta * loss_t).
        # `-np.min(loss_t)` is applied only to avoid entries being near zero for large eta*loss_t values before the normalization
        posterior_t  /= np.sum(posterior_t)  # normalize

        # Log posterior_t
        posterior_t_log[t-1, :] = posterior_t

        # Compute u_t
        u_t = _compute_u_t(data, posterior_t, predictions[t-1, :], bias_scale)

        # Sanity checks for sampling probability
        if u_t > 1:
            u_t = 1
        if np.logical_and(u_t>=0, u_t<=1):
            u_t = u_t
        else:
            u_t = 0

        # Is x_t in the region of disagreement? yes if dis_t>1, no otherwise
        dist_t = len(np.unique(predictions[t-1, :]))

        # If u_t is in the region of agreement, don't query anything
        if dist_t == 1:
            u_t = 0
            z_t = 0
            z_t_log[t-1] = z_t
        else:
            #Else, make a random query decision
            if u_t>0:
                u_t = np.maximum(u_t, eta)
            if u_t>1:
                u_t=1
            z_t = np.random.binomial(size=1, n=1, p=u_t)
            z_t_log[t-1] = z_t

        if z_t == 1:
            loss_t += (np.array((predictions[t-1, :] != oracle[t-1]) * 1) / u_t)
            loss_t = loss_t.reshape(data._num_models, 1)
            loss_t = np.squeeze(np.asarray(loss_t))

        m_star = np.random.choice(list(range(data._num_models)), p=posterior_t)
        # Incur hidden loss
        hidden_loss_log[t-1] = (predictions[t-1, m_star] != oracle[t-1]) * 1
        # print(z_t)
        # print(loss_t)

        # Terminate if it exceeds the budget
        if np.sum(z_t_log) < budget:
            z_t_budget[t-1] = z_t_log[t-1]


    # Labelling decisions as 0's and 1's
    labelled_instances = z_t_log
    ct_log = np.ones(data._num_instances, dtype=int)


    return (labelled_instances, ct_log, z_t_budget, hidden_loss_log, posterior_t_log)


##

def _compute_u_t(data, posterior_t, predictions_c, bias_scale):

    # Initialize possible u_t's
    u_t_list = np.zeros(data._num_classes)

    # Repeat for each class
    for c in range(data._num_classes):
        # Compute the loss of models if the label of the streamed data is "c"
        loss_c = np.array(predictions_c != c)*1
        #
        # Compute the respective u_t value (conditioned on class c)
        term1 = np.inner(posterior_t, loss_c)
        u_t_list[c] = term1*(1-term1)

    # Return the final u_t
    u_t = bias_scale * np.max(u_t_list)

    return u_t