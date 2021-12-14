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

    # Initializations of hyperparameters
    eta_t = np.sqrt(np.log(num_models)/2) # initialize the hyperparameter
    bias_scale = 1 # this linear scaling parameter is added to boost the coin tossing bias in case needed
    cost = 0 # keep record of how many instances are queried

    # Shuffle the indices to reduce time-dependency
    shuffled_indices = np.random.permutation(num_instances)
    predictions = predictions[shuffled_indices, :]


    # Initialization of posterior belief and momentary loss
    loss_t = np.zeros(num_models) # loss per models
    posterior_t = np.ones(num_models)/num_models


    while cost <= budget: # repeat while budget is not exceeded
        # For each streaming data instance
        for t in np.arange(1, num_instances+1, 1):

            # Edit eta
            eta_t = eta_t / np.sqrt(t)


            posterior_t = np.exp(-eta_t * (loss_t - np.min(loss_t)))
            # Note that above equation is equivalent to np.exp(-eta * loss_t).
            # `-np.min(loss_t)` is applied only to avoid entries being near zero for large eta*loss_t values before the normalization
            posterior_t  /= np.sum(posterior_t)  # normalize

            ### Toss a coin if xt is in the region of disagreement, else skip
            if len(np.unique(predictions[t - 1, :])) == 1:
                zt = 0
            else:
                (zt, ut) = _coin_tossing(predictions, posterior_t, labelset, bias_scale)

            # Update the cost
            cost += zt

            # If the coin is HEADS, query the label and update the posterior. Else no update necessary
            if zt == 1:
                print("Please enter the label for instance with ID "+str(shuffled_indices[t-1])+":")
                label_t = input()
                loss_t += (np.array((predictions[t-1, :] != label_t) * 1) / ut)
                loss_t = loss_t.reshape(num_models, 1)
                loss_t = np.squeeze(np.asarray(loss_t))

    bestmodel = np.argmax(posterior_t)

    return (bestmodel, posterior_t)


###

def _coin_tossing(pred, post, labelset, scale):

    ### Compute ut

    # Initialize possible u_t's
    num_classes = len(labelset)
    ut_list = np.zeros(num_classes)

    # Repeat for each class
    for c in labelset:
        # Compute the loss of models if the label of the streamed data is "c"
        loss_c = np.array(pred != c)*1
        #
        # Compute the respective u_t value (conditioned on class c)
        innprod = np.inner(post, loss_c)
        ut_list[c] = innprod*(1-innprod)

    # Compute the final ut
    ut = scale * np.max(ut_list)
    # Toss the coin
    zt = np.random.binomial(size=1, n=1, p=ut)

    return(zt, ut)