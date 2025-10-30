import torch
import torch.nn as nn


class GradNormLoss(nn.Module):
    
    def __init__(self, num_tasks: int, alpha=1.5):
        super().__init__()

        self.num_tasks = num_tasks
        self.alpha = alpha

        self.loss_weights = self.to_device(nn.Parameter(torch.ones(num_tasks)))
        self.initial_losses = None


    def to_device(self, obj):
        if torch.cuda.is_available():
            obj = obj.to("cuda")
        elif torch.backends.mps.is_available():
            obj = obj.to("mps")

        return obj


    def forward(self, losses: torch.Tensor, layer_weights):
        """Regular forward pass that calculates the combined loss
        args:
            losses: torch tensor of loss values

        returns:
            combined_loss_sum: a combined loss value after weighting
        """

        # Record initial loss values
        # Maybe set these to a default value of log(C)?
        if self.initial_losses is None:
            self.initial_losses = losses.detach() # Detach to prevent gradients from flowing

        # Calculate combined weighted loss function
        combined_loss = losses * self.loss_weights

        combined_loss_sum = combined_loss.sum()

        # Compute Lgrad loss
        Lgrad = self.compute_gradnorm_loss(losses, layer_weights)

        # Compute gradients for Lgrad
        Lgrad.backward()

        # Compute standard gradients 
        # combined_loss_sum.backward(retain_graph=False)



        # Renormalize loss weights so that they add up to num_tasks
        weights_sum = torch.sum(self.loss_weights)
        scale_factor = self.num_tasks / weights_sum

        self.loss_weights *= scale_factor

        return combined_loss_sum


    def compute_gradnorm_loss(self, losses: torch.Tensor, layer_weights):
        
        # Compute l2 norms of the gradients of each task's loss G_W^i(t)
        Gws = []
        for i in range(self.num_tasks):
            weighted_loss = self.loss_weights[i] * losses[i]

            # Compute gradient of weighted loss wrt weights W
            weighted_loss_gradient = torch.autograd.grad(weighted_loss, layer_weights, 
                                                         retain_graph=True, create_graph=True)

            # Take L2 norm of gradient
            Gwi = torch.linalg.vector_norm(weighted_loss_gradient[0])

            Gws.append(Gwi)

        # Convert to torch tensor
        Gws = torch.stack(Gws)

        # Compute ri(t) = Li(t) / Etask[Li(t)]
        ris = []
        # Compute Li(t)'s
        for i in range(self.num_tasks):
            # Calculate loss ratio
            loss_ratio = losses[i] / self.initial_losses[i]

            ris.append(loss_ratio)

        # Compute average loss ratio Etask[Li(t)]
        ris = torch.tensor(ris, device='mps') # TODO: devices bruh
        average_loss_ratio = torch.mean(ris)

        # Divide by average loss ratio to get ri(t)'s
        ris /= average_loss_ratio

        # Compute the average gradient norm G_W(t) bar
        average_gradient_norm = torch.sum(Gws) / self.num_tasks

        # Make sure the target gradnorm value is treated as a constant
        target = average_gradient_norm * ris ** self.alpha
        # Compute GradNorm loss L_grad
        Lgrad = torch.sum(torch.abs(Gws - target))

        return Lgrad


