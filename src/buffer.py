import random
import torch


class LatentReservoirBuffer:
    """
    Custom reservoir buffer class for batches of samples without labels, but with encoder features.
    """
    def __init__(self, buffer_size, alpha_ema=1.0, device='cpu'):
        self.buffer_size = buffer_size # Maximum size of the buffer
        self.buffer_features = torch.empty(0,1).to(device) # Buffer for corresponding sample features
        self.buffer_labels = torch.empty(0, 1).to(device) # Buffer for corresponding sample labels
        self.alpha_ema = alpha_ema # 1.0 = do not update stored features, 0.0 = substitute with new features
        self.device = device

        self.seen_samples = 0 # Samples seen so far

    # Add a batch of samples and features to the buffer
    def add(self, batch_features, batch_labels):
        assert batch_features.size(0) == batch_labels.size(0)

        batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)

        # Initialize empty buffers
        if self.buffer_features.size(0) == 0:

            # Extend buffer_features to have same dim of batch_features
            buffer_shape = list(batch_features.size())
            buffer_shape[0] = 0
            self.buffer_features = torch.empty(buffer_shape).to(self.device)

            # Extend buffer_labels to have same dim of batch_labels
            buffer_shape = list(batch_labels.size())
            buffer_shape[0] = 0
            self.buffer_labels = torch.empty(buffer_shape, dtype=batch_labels.dtype).to(self.device)

        batch_size = batch_features.size(0)

        if self.seen_samples < self.buffer_size:
            # Store samples until the buffer is full
            if self.seen_samples + batch_size <= self.buffer_size:
                # If there is enough space in the buffer, add all the samples
                self.buffer_features = torch.cat((self.buffer_features, batch_features), dim=0)
                self.buffer_labels = torch.cat((self.buffer_labels, batch_labels), dim=0)
                self.seen_samples += batch_size
            else:
                # If there is not enough space, add only the remaining samples
                remaining_space = self.buffer_size - self.seen_samples
                self.buffer_features = torch.cat((self.buffer_features, batch_features[:remaining_space]), dim=0)
                self.buffer_labels = torch.cat((self.buffer_labels, batch_labels[:remaining_space]), dim=0)
                self.seen_samples += remaining_space
        else:
            # Replace samples with probability buffer_size/seen_samples
            for i in range(batch_size):
                replace_index = random.randint(0, self.seen_samples + i)

                if replace_index < self.buffer_size:
                    self.buffer_features[replace_index] = batch_features[i]
                    self.buffer_labels[replace_index] = batch_labels[i]
            
            self.seen_samples += batch_size

    # Sample batch_size samples from the buffer, 
    # returns samples and indices of extracted samples (for feature update)
    def sample(self, batch_size):
        assert batch_size <= len(self.buffer_features)

        # Sample batch_size indices
        indices = random.sample(range(len(self.buffer_features)), batch_size)

        # Get sample batch from indices
        batch_features = self.buffer_features[indices]
        batch_labels = self.buffer_labels[indices]

        return batch_features, batch_labels, indices
    
    # Update features of buffer samples at given indices
    def update_features(self, batch_features, indices):
        assert batch_features.size(0) == len(indices)

        batch_features = batch_features.to(self.device)

        for i, idx in enumerate(indices):
            if self.buffer_features[idx] is not None:
                # There are already features stored for that sample
                # EMA update of features
                self.buffer_features[idx] = self.alpha_ema * self.buffer_features[idx] + (1 - self.alpha_ema) * batch_features[i]
            else:
                # No features stored yet, store newly passed features
                self.buffer_features[idx] = batch_features[i]

    # Sample given indices 
    def sample_indices(self, indices):
        return self.buffer_features[indices], self.buffer_labels[indices]