import numpy as np

def accuracy(outputs, labels):
    '''
    Compute the accuracy, given the outputs and labels for all tokens. Exclude
    padding terms.

    Parameters
    ----------
    outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log
        softmax output of the model
    labels: (np.ndarray) dimension batch_size x seq_len where each element is
        either a label in [0, 1, ... num_tag-1], or -1 in case it is a padding
        token.

    Returns
    -------
    a float accuracy in range [0,1]
    '''
    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.ravel()

    # since padding tokens have label -1, we can generate a mask to exclude
    # the loss from those terms
    mask = (labels >= 0)

    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)

    # compare outputs with labels and divide by number of tokens (excluding
    # padding tokens)
    return np.sum(outputs==labels)/float(np.sum(mask))


def f_beta_score(outputs, labels, beta):
    '''
    Compute the f-beta score, given the outputs and labels for all tokens.
    Exclude padding terms.

    Parameters
    ----------
    outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log
        softmax output of the model
    labels: (np.ndarray) dimension batch_size x seq_len where each element is
        either a label in [0, 1, ... num_tag-1], or -1 in case it is a padding
        token.
    beta: (int or float) the weight of precision in the combined harmonic mean
        score between precision and recall. beta < 1 lends more weight to
        precision, while beta > 1 favors recall.

    Returns
    -------
    a float f-beta score in range [0,1]
    '''
    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.ravel()

    # since padding tokens have label -1, we can generate a mask to exclude
    # the loss from those terms
    mask = (labels >= 0)

    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)

    # compute true positive, false positive, and true negative rates
    tp = np.sum(np.logical_and(outputs == 1, labels == 1))
    fp = np.sum(np.logical_and(outputs == 1, labels == 0))
    fn = np.sum(np.logical_and(outputs == 0, labels == 1))

    # compute f-beta score using tp, fp, and tn
    coeff = (1 + beta ** 2)
    f_beta = coeff * tp /(coeff * tp + beta ** 2 * fn + fp)
    
    return f_beta


def f1_score(outputs, labels):
    '''
    Compute the f-1 score, given the outputs and labels for all tokens.
    Exclude padding terms.

    Parameters
    ----------
    outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log
        softmax output of the model
    labels: (np.ndarray) dimension batch_size x seq_len where each element is
        either a label in [0, 1, ... num_tag-1], or -1 in case it is a padding
        token.

    Returns
    -------
    a float f-1 score in range [0,1]
    '''
    return f_beta_score(outputs, labels, beta=1)


def f2_score(outputs, labels):
    '''
    Compute the f-2 score, given the outputs and labels for all tokens.
    Exclude padding terms.

    Parameters
    ----------
    outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log
        softmax output of the model
    labels: (np.ndarray) dimension batch_size x seq_len where each element is
        either a label in [0, 1, ... num_tag-1], or -1 in case it is a padding
        token.

    Returns
    -------
    a float f-2 score in range [0,1]
    '''
    return f_beta_score(outputs, labels, beta=2)
