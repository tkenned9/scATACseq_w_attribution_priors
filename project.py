import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special
import scipy.stats
import scipy.ndimage
import sklearn.metrics
import pyfaidx
import pyBigWig

CUDA_VISIBLE_DEVICES=1,2

# From https://github.com/amtseng/fourier_attribution_priors/blob/master/notebooks/fourier_prior_example.ipynb
def dna_to_one_hot(seqs):
    """
    Converts a list of DNA ("ACGT") sequences to one-hot encodings, where the
    position of 1s is ordered alphabetically by "ACGT". `seqs` must be a list
    of N strings, where every string is the same length L. Returns an N x L x 4
    NumPy array of one-hot encodings, in the same order as the input sequences.
    All bases will be converted to upper-case prior to performing the encoding.
    Any bases that are not "ACGT" will be given an encoding of all 0s.
    """
    seq_len = len(seqs[0])
    assert np.all(np.array([len(s) for s in seqs]) == seq_len)

    # Join all sequences together into one long string, all uppercase
    seq_concat = "".join(seqs).upper()

    one_hot_map = np.identity(5)[:, :-1]

    # Convert string into array of ASCII character codes;
    base_vals = np.frombuffer(bytearray(seq_concat, "utf8"), dtype=np.int8)

    # Anything that's not an A, C, G, or T gets assigned a higher code
    base_vals[~np.isin(base_vals, np.array([65, 67, 71, 84]))] = 85

    # Convert the codes into indices in [0, 4], in ascending order by code
    _, base_inds = np.unique(base_vals, return_inverse=True)

    # Get the one-hot encoding for those indices, and reshape back to separate
    return one_hot_map[base_inds].reshape((len(seqs), seq_len, 4))

# From https://github.com/amtseng/fourier_attribution_priors/blob/master/notebooks/fourier_prior_example.ipynb
def one_hot_to_dna(one_hot):
    """
    Converts a one-hot encoding into a list of DNA ("ACGT") sequences, where the
    position of 1s is ordered alphabetically by "ACGT". `one_hot` must be an
    N x L x 4 array of one-hot encodings. Returns a lits of N "ACGT" strings,
    each of length L, in the same order as the input array. The returned
    sequences will only consist of letters "A", "C", "G", "T", or "N" (all
    upper-case). Any encodings that are all 0s will be translated to "N".
    """
    bases = np.array(["A", "C", "G", "T", "N"])
    # Create N x L array of all 5s
    one_hot_inds = np.tile(one_hot.shape[2], one_hot.shape[:2])

    # Get indices of where the 1s are
    batch_inds, seq_inds, base_inds = np.where(one_hot)

    # In each of the locations in the N x L array, fill in the location of the 1
    one_hot_inds[batch_inds, seq_inds] = base_inds

    # Fetch the corresponding base for each position using indexing
    seq_array = bases[one_hot_inds]
    return ["".join(seq) for seq in seq_array]

# Adapted from https://github.com/amtseng/fourier_attribution_priors/blob/master/notebooks/fourier_prior_example.ipynb
def place_tensor(tensor, cuda = 0):
    """
    Places a tensor on GPU, if PyTorch sees CUDA; otherwise, the returned tensor
    remains on CPU.
    """
    if torch.cuda.is_available():
        return tensor.cuda(cuda)
    return tensor

# Adapted from https://github.com/amtseng/fourier_attribution_priors/blob/master/notebooks/fourier_prior_example.ipynb
def smooth_tensor_1d(input_tensor, smooth_sigma, cuda = 0):
    """
    Smooths an input tensor along a dimension using a Gaussian filter.
    Arguments:
        `input_tensor`: a A x B tensor to smooth along the second dimension
        `smooth_sigma`: width of the Gaussian to use for smoothing; this is the
            standard deviation of the Gaussian to use, and the Gaussian will be
            truncated after 1 sigma (i.e. the smoothing window is
            1 + (2 * sigma); sigma of 0 means no smoothing
    Returns an array the same shape as the input tensor, with the dimension of
    `B` smoothed.
    """
    # Generate the kernel
    if smooth_sigma == 0:
        sigma, truncate = 1, 0
    else:
        sigma, truncate = smooth_sigma, 1
    base = np.zeros(1 + (2 * sigma))
    base[sigma] = 1  # Center of window is 1 everywhere else is 0
    kernel = scipy.ndimage.gaussian_filter(base, sigma=sigma, truncate=truncate)
    kernel = place_tensor(torch.tensor(kernel), cuda)

    # Expand the input and kernel to 3D, with channels of 1
    # Also make the kernel float-type, as the input is going to be of type float
    input_tensor = torch.unsqueeze(input_tensor, dim=1)
    kernel = torch.unsqueeze(torch.unsqueeze(kernel, dim=0), dim=1).float()

    smoothed = torch.nn.functional.conv1d(
        input_tensor, kernel, padding=sigma
    )

    return torch.squeeze(smoothed, dim=1)


# From https://gist.github.com/amtseng/1d3cb433aef2b2a1e601346ff4fc8e96#file-sc_atac_arch-py
def multinomial_log_probs(category_log_probs, trials, query_counts):
    """
    Defines multinomial distributions and computes the probability of seeing
    the queried counts under these distributions. This defines D different
    distributions (that all have the same number of classes), and returns D
    probabilities corresponding to each distribution.
    Arguments:
        `category_log_probs`: a D x N tensor containing log probabilities (base
            e) of seeing each of the N classes/categories
        `trials`: a D-tensor containing the total number of trials for each
            distribution (can be different numbers)
        `query_counts`: a D x N tensor containing the observed count of each
            category in each distribution; the probability is computed for these
            observations
    Returns a D-tensor containing the log probabilities (base e) of each
    observed query with its corresponding distribution. Note that D can be
    replaced with any shape (i.e. only the last dimension is reduced).
    """
    # Multinomial probability = n! / (x1!...xk!) * p1^x1 * ... pk^xk
    # Log prob = log(n!) - (log(x1!) ... + log(xk!)) + x1log(p1) ... + xklog(pk)
    trials, query_counts = trials.float(), query_counts.float()
    log_n_fact = torch.lgamma(trials + 1)
    log_counts_fact = torch.lgamma(query_counts + 1)
    log_counts_fact_sum = torch.sum(log_counts_fact, dim=-1)
    log_prob_pows = category_log_probs * query_counts  # Elementwise sum
    log_prob_pows_sum = torch.sum(log_prob_pows, dim=-1)

    return log_n_fact - log_counts_fact_sum + log_prob_pows_sum

# From https://gist.github.com/amtseng/1d3cb433aef2b2a1e601346ff4fc8e96#file-sc_atac_arch-py
def profile_logits_to_log_probs(logit_pred_profs, axis=2):
    """
    Converts the model's predicted profile logits into normalized probabilities
    via a softmax on the specified dimension (defaults to axis=2).
    Arguments:
        `logit_pred_profs`: a tensor/array containing the predicted profile
            logits
    Returns a tensor/array of the same shape, containing the predicted profiles
    as log probabilities by doing a log softmax on the specified dimension. If
    the input is a tensor, the output will be a tensor. If the input is a NumPy
    array, the output will be a NumPy array. Note that the  reason why this
    function returns log probabilities rather than raw probabilities is for
    numerical stability.
    """
    if type(logit_pred_profs) is np.ndarray:
        return logit_pred_profs - \
            scipy.special.logsumexp(logit_pred_profs, axis=axis, keepdims=True)
    else:
        return torch.log_softmax(logit_pred_profs, dim=axis)

    
def save_model(model, save_path):
    """
    Saves the given model at the given path. This saves the state of the model
    (i.e. trained layers and parameters), and the arguments used to create the
    model (i.e. a dictionary of the original arguments).
    """
    save_dict = {
        "model_state": model.state_dict()
    }
    torch.save(save_dict, save_path)


def restore_model(load_path, cuda=0):
    """
    Restores a model from the given path. It will then restore the learned
    parameters to the model.
    """
    load_dict = torch.load(load_path)
    model_state = load_dict["model_state"]
    model = ProfilePredictor(cuda=cuda)
    model.load_state_dict(model_state)
    return model


class DataLoader:
    def __init__(
        self, data_npy_path, batch_size,
        reference_genome_path='./data/hg38.fasta', reads_path='./data/vcm_reads.bw',
        bp_expansion=346, reverse_complement=False, seed=242, jitter_seed=12312
    ):
        data_table = pd.DataFrame(data=np.load(data_npy_path, allow_pickle=True),
                            columns=["chrom", "start", "end"])


        self.coords = data_table.values
        self.batch_size = batch_size
        self.reference_genome_path = reference_genome_path
        self.reads = pyBigWig.open(reads_path)
        self.bp_expansion = bp_expansion
        self.shuffle_rng = np.random.RandomState(seed)
        self.jitter_rng = np.random.RandomState(jitter_seed)
        self.reverse_complement = reverse_complement

    def shuffle_data(self):
        perm = self.shuffle_rng.permutation(len(self.coords))
        self.coords = self.coords[perm]

    def __len__(self):
        return int(np.ceil(len(self.coords) / self.batch_size))

    def __getitem__(self, index):
        batch_slice = slice(index * self.batch_size, (index + 1) * self.batch_size)
        batch_coords = self.coords[batch_slice].copy()
        jitter = self.jitter_rng.randint(-256,257, batch_coords.shape[0])

        for i in range(batch_coords.shape[0]):
            batch_coords[i][1] += jitter[i]
            batch_coords[i][2] += jitter[i]


        batch_profiles = np.array([self.reads.values(region[0], region[1], region[2]) for region in batch_coords])
        batch_profiles[np.isnan(batch_profiles)] = 0

        for i in range(batch_coords.shape[0]):
            batch_coords[i][1] -= int(self.bp_expansion/2)
            batch_coords[i][2] += int(self.bp_expansion/2)


        genome_reader = pyfaidx.Fasta(self.reference_genome_path)
        seqs = [
            genome_reader[chrom][start:end].seq for
            chrom, start, end in batch_coords
        ]

        one_hot = dna_to_one_hot(seqs)

        if not self.reverse_complement:
            return one_hot, batch_profiles
        else:
            return np.concatenate([one_hot, np.flip(one_hot, axis=(1, 2))]), \
                np.concatenate([batch_profiles, np.flip(batch_profiles, axis=1)])


# Adapted from https://gist.github.com/amtseng/1d3cb433aef2b2a1e601346ff4fc8e96#file-sc_atac_arch-py-L189
class ProfilePredictor(torch.nn.Module): 
    def __init__(self, input_length=1346, input_depth=4, profile_length=1000,
                num_tasks=1, num_strands=1, num_dil_conv_layers=7,
                dil_conv_filter_sizes= ([21] + (6*[3])), dil_conv_stride=1,
                dil_conv_dilations=[2**i for i in range(7)], dil_conv_depths=([256]*7),
                prof_conv_kernel_size=75, prof_conv_stride=1, cuda=0):

        """
        Creates a profile predictor from a DNA sequence that does not take
        control profiles.
        Arguments:
            `input_length`: length of the input sequences; each input sequence
                would be D x L, where L is the length
            `input_depth`: depth of the input sequences; each input sequence
                would be D x L, where D is the depth
            `profile_length`: length of the predicted profiles; it must be
                consistent with the convolutional layers specified
            `num_tasks`: number of tasks that are to be predicted
            `num_strands`: number of strands for each profile, typically 1 or 2
            `num_dil_conv_layers`: number of dilating convolutional layers
            `dil_conv_filter_sizes`: sizes of the initial dilating convolutional
                filters; must have `num_conv_layers` entries
            `dil_conv_stride`: stride used for each dilating convolution
            `dil_conv_dilations`: dilations used for each layer of the dilating
                convolutional layers
            `dil_conv_depths`: depths of the dilating convolutional filters;
                must have `num_conv_layers` entries
            `prof_conv_kernel_size`: size of the large convolutional filter used
                for profile prediction
            `prof_conv_stride`: stride used for the large profile convolution
        Creates a close variant of the BPNet architecture, as described here:
            https://www.biorxiv.org/content/10.1101/737981v1.full
        """

        super().__init__()

        assert len(dil_conv_filter_sizes) == num_dil_conv_layers
        assert len(dil_conv_dilations) == num_dil_conv_layers
        assert len(dil_conv_depths) == num_dil_conv_layers
        
        self.gpu = cuda
        self.input_depth = input_depth
        self.input_length = input_length
        self.profile_length = profile_length
        self.num_tasks = num_tasks
        self.num_strands = num_strands
        self.num_dil_conv_layers = num_dil_conv_layers
        self.mse_loss = torch.nn.MSELoss(reduction="none")

        ## Model Architecture

        ## Dilated convolutional layers
        self.dil_convs = torch.nn.ModuleList()
        last_out_size = input_length # will keep track of output size if there was no padding
        for i in range(num_dil_conv_layers):
            kernel_size = dil_conv_filter_sizes[i]
            in_channels = input_depth if i == 0 else dil_conv_depths[i-1]
            out_channels = dil_conv_depths[i]
            dilation = dil_conv_dilations[i]
            padding = int(dilation * (kernel_size - 1) / 2) # "same" padding
            self.dil_convs.append(
                torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, dilation=dilation, padding=padding))
            last_out_size = last_out_size - (dilation * (kernel_size - 1)) #hypothetical decrease

        # This is used to truncate the output of dilated layers in forward
        self.last_dil_conv_size = last_out_size

        # ReLU activation for the convolutional layers and attribution prior
        self.relu = torch.nn.ReLU()

        # Architecture specific to profile distribution prediction:
        # Conv layer with large kernel
        self.prof_large_conv = torch.nn.Conv1d(
            in_channels = dil_conv_depths[-1], # 256
            out_channels=(num_tasks*num_strands), # 1
            kernel_size=prof_conv_kernel_size
        )

        self.prof_pred_size = self.last_dil_conv_size - (prof_conv_kernel_size - 1)
        assert self.prof_pred_size == profile_length, \
            "Prediction length is specified to be %d, but with the given " +\
            "input length of %d and the given convolutions, the computed " +\
            "prediction length is %d" % \
            (profile_length, input_length, self.prof_pred_size)

        # Length-1 convolution over the convolutional output to get the final profile
        self.prof_one_conv = torch.nn.Conv1d(
            in_channels=(num_tasks * num_strands), # 1
            out_channels=(num_tasks * num_strands), # 1
            kernel_size=1, groups=num_tasks  # One set of filters over each task
        )

        # Architecture specific to counts prediction
        # Global average pooling, outputs 1 number for each of 256 channels
        self.count_pool = torch.nn.AvgPool1d(kernel_size=self.last_dil_conv_size)

        # Dense layer to consolidate pooled result to small number of features
        self.count_dense = torch.nn.Linear(
            in_features=dil_conv_depths[-1], #256
            out_features=(num_tasks * num_strands) #1
        )

        # Dense layer over pooling features to get the final counts, implemented
        # as grouped convolution with kernel size 1. In our case just scaling one number
        self.count_one_conv = torch.nn.Conv1d(
            in_channels=(num_tasks * num_strands), # 1
            out_channels=(num_tasks * num_strands), # 1
            kernel_size=1, groups=num_tasks #1
        )


    def forward(self, input_seqs, cont_profs=None):
        """
        Computes a forward pass on a batch of sequences.
        Arguments:
            `inputs_seqs`: a B x I x D tensor, where B is the batch size, I is
                the input sequence length, and D is the number of input channels
            `cont_profs`: unused parameter, existing only for compatibility
        Returns the predicted profiles (unnormalized logits) for each task and
        each strand (a B x T x O x S tensor), and the predicted log
        counts (base e) for each task and each strand (a B x T x S) tensor.
        """
        
        # input seqs: batch_size x 1346 x 4
        # will output (batch_size x 1000) tensor for profiles
        # and (batch_size x 1) tensor for log_e counts
        
        batch_size = input_seqs.size(0)
        input_length = input_seqs.size(1)
        assert input_length == self.input_length

        # PyTorch prefers convolutions to be channel first, so transpose the input
        input_seqs = input_seqs.transpose(1, 2)  # Shape is now (batch_size x 4 x 1346)

        # Dilated convs where each layers input is sum of previous layers
        dil_conv_out_list = None
        dil_conv_sum = 0

        for i, dil_conv in enumerate(self.dil_convs):
            if i == 0:
                dil_conv_out = self.relu(dil_conv(input_seqs))
            else:
                dil_conv_out = self.relu(dil_conv(dil_conv_sum))

            if i != self.num_dil_conv_layers - 1:
                dil_conv_sum = dil_conv_out + dil_conv_sum


        # Truncate the final dilated convolutional layer output so that it
        # only has entries that did not see padding; this is equivalent to
        # truncating it to the size it would be if no padding were ever added
        start = int((dil_conv_out.size(2) - self.last_dil_conv_size)/2)
        end = start + self.last_dil_conv_size
        dil_conv_out_cut = dil_conv_out[:, :, start : end]

        # Profile distribution prediction:
        # Perform convolution with a large kernel
        prof_large_conv_out = self.prof_large_conv(dil_conv_out_cut)
        # output shape: (batch_size x 1000)

        # Length 1 convolution
        prof_one_conv_out = self.prof_one_conv(prof_large_conv_out)
        # output shape: batch_size x 1


        prof_pred = prof_one_conv_out.view(
            batch_size, self.num_tasks, self.num_strands, -1
        )
        # Transpose profile predictions to get B x T x O x S
        prof_pred = prof_pred.transpose(2, 3)


        # Count Prediction:
        # Global Average Pooling
        count_pool_out = self.count_pool(dil_conv_out_cut)  # Shape: B x 256 x 1
        count_pool_out = torch.squeeze(count_pool_out, dim=2)

        # Reduce pooling output to fewer features, a pair for each task
        count_dense_out = self.count_dense(count_pool_out)  # Shape: B x ST
        count_dense_out = count_dense_out.view(
            batch_size, self.num_strands * self.num_tasks, 1
        )

        # Dense layer over the last layer's outputs; each set of counts gets
        # a different dense network (implemented as convolution with kernel size
        # 1)
        count_one_conv_out = self.count_one_conv(count_dense_out)
        # Shape: B x ST x 1
        count_pred = count_one_conv_out.view(
            batch_size, self.num_tasks, self.num_strands, -1
        )
        # Shape: B x T x S x 1
        count_pred = torch.squeeze(count_pred, dim=3)  # Shape: B x T x S

        return prof_pred, count_pred

    # loss functions from: https://gist.github.com/amtseng/1d3cb433aef2b2a1e601346ff4fc8e96#file-sc_atac_arch-py
    def correctness_loss(self, true_profs, logit_pred_profs, log_pred_counts,
                         count_loss_weight,return_separate_losses=False):
        """
        Returns the loss of the correctness of the predicted profiles and
        predicted read counts. This prediction correctness loss is split into a
        profile loss and a count loss. The profile loss is the -log probability
        of seeing the true profile read counts, given the multinomial
        distribution defined by the predicted profile count probabilities. The
        count loss is a simple mean squared error on the log counts.
        Arguments:
            `true_profs`: a B x T x O x S tensor containing true UNnormalized
                profile values, where B is the batch size, T is the number of
                tasks, O is the profile length, and S is the number of strands;
                the sum of a profile gives the raw read count for that task
            `logit_pred_profs`: a B x T x O x S tensor containing the predicted
                profile _logits_
            `log_pred_counts`: a B x T x S tensor containing the predicted log
                read counts (base e)
            `count_loss_weight`: amount to weight the portion of the loss for
                the counts
            `return_separate_losses`: if True, also return the profile and
                counts losses (scalar Tensors)
        Returns a scalar loss tensor, or perhaps 3 scalar loss tensors.
        """

        assert true_profs.size() == logit_pred_profs.size()
        batch_size = true_profs.size(0)
        num_tasks = true_profs.size(1)
        num_strands = true_profs.size(3)

        # Add the profiles together to get the raw counts
        true_counts = torch.sum(true_profs, dim=2)  # Shape: B x T x S

        # Transpose and reshape the profile inputs from B x T x O x S to
        # B x ST x O; all metrics will be computed for each individual profile,
        # then averaged across pooled tasks/strands, then across the batch
        true_profs = true_profs.transpose(2, 3).reshape(
            batch_size, num_tasks * num_strands, -1
        )
        logit_pred_profs = logit_pred_profs.transpose(2, 3).reshape(
            batch_size, num_tasks * num_strands, -1
        )
        # Reshape the counts from B x T x S to B x ST
        true_counts = true_counts.view(batch_size, num_tasks * num_strands)
        log_pred_counts = log_pred_counts.view(
            batch_size, num_tasks * num_strands
        )

        # 1. Profile loss
        # Compute the log probabilities based on multinomial distributions,
        # each one is based on predicted probabilities, one for each track

        # Convert logits to log probabilities (along the O dimension)
        log_pred_profs = profile_logits_to_log_probs(logit_pred_profs, axis=2)

        # Compute probability of seeing true profile under distribution of log
        # predicted probs
        neg_log_likelihood = -multinomial_log_probs(
            log_pred_profs, true_counts, true_profs
        )  # Shape: B x 2T
        # Average across tasks/strands, and then across the batch
        batch_prof_loss = torch.mean(neg_log_likelihood, dim=1)
        prof_loss = torch.mean(batch_prof_loss)

        # 2. Counts loss
        # Mean squared error on the log counts (with 1 added for stability)
        log_true_counts = torch.log(true_counts + 1)
        mse = self.mse_loss(log_pred_counts, log_true_counts)

        # Average across tasks/strands, and then across the batch
        batch_count_loss = torch.mean(mse, dim=1)
        count_loss = torch.mean(batch_count_loss)

        final_loss = prof_loss + (count_loss_weight * count_loss)

        if return_separate_losses:
            return final_loss, prof_loss, count_loss
        else:
            return final_loss


    def fourier_att_prior_loss(
        self, status, input_grads, freq_limit, limit_softness,
        att_prior_grad_smooth_sigma):
        """
        Computes an attribution prior loss for some given training examples,
        using a Fourier transform form.
        Arguments:
            `status`: a B-tensor, where B is the batch size; each entry is 1 if
                that example is to be treated as a positive example, and 0
                otherwise
            `input_grads`: a B x L x D tensor, where B is the batch size, L is
                the length of the input, and D is the dimensionality of each
                input base; this needs to be the gradients of the input with
                respect to the output (for multiple tasks, this gradient needs
                to be aggregated); this should be *gradient times input*
            `freq_limit`: the maximum integer frequency index, k, to consider
                for the loss; this corresponds to a frequency cut-off of
                pi * k / L; k should be less than L / 2
            `limit_softness`: amount to soften the limit by, using a hill
                function; None means no softness
            `att_prior_grad_smooth_sigma`: amount to smooth the gradient before
                computing the loss
        Returns a single scalar Tensor consisting of the attribution loss for
        the batch.
        """
        abs_grads = torch.sum(torch.abs(input_grads), dim=2)

        # Smooth the gradients
        grads_smooth = smooth_tensor_1d(
            abs_grads, att_prior_grad_smooth_sigma, self.gpu
        )

        # Only do the positives
        pos_grads = grads_smooth[status == 1]

        # Loss for positives
        if pos_grads.nelement():
            pos_fft = torch.rfft(pos_grads, 1)
            pos_mags = torch.norm(pos_fft, dim=2)
            pos_mag_sum = torch.sum(pos_mags, dim=1, keepdim=True)
            pos_mag_sum[pos_mag_sum == 0] = 1  # Keep 0s when the sum is 0
            pos_mags = pos_mags / pos_mag_sum

            # Cut off DC
            pos_mags = pos_mags[:, 1:]

            # Construct weight vector
            weights = place_tensor(torch.ones_like(pos_mags), self.gpu)
            if limit_softness is None:
                weights[:, freq_limit:] = 0
            else:
                x = place_tensor(
                    torch.arange(1, pos_mags.size(1) - freq_limit + 1), self.gpu
                ).float()
                weights[:, freq_limit:] = 1 / (1 + torch.pow(x, limit_softness))

            # Multiply frequency magnitudes by weights
            pos_weighted_mags = pos_mags * weights

            # Add up along frequency axis to get score
            pos_score = torch.sum(pos_weighted_mags, dim=1)
            pos_loss = 1 - pos_score
            return torch.mean(pos_loss)
        else:
            return place_tensor(torch.zeros(1), self.gpu)
