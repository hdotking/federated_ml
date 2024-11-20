from typing import List, Tuple

import torch
from torch import nn, optim

"""
NER with Federated Learning

**How do we solve the problem that the Entities themselves (the vocabulary) isn't shared among clients?**

If a pre-existing vocabulary public is insufficient or unavailable, an alternative is to create a global vocabulary
through secure federated techniques that ensure privacy. Here are some methods to do this:

**Federated Frequency Counting with Secure Aggregation:**

- Each client builds a local vocabulary from its own data and computes token frequency counts.
- These token counts are aggregated across clients using secure aggregation techniques
  (e.g., **homomorphic encryption or secure multi-party computation**).
  This allows the central server to learn which tokens are most frequent across all clients without ever seeing
  individual tokens or local data.
- The server can then build a global vocabulary based on the aggregated counts while ensuring that no client
  data is exposed.
"""


class SimpleNN(torch.nn.Module):
    """
    If we swap for an a LLM then be sure to change batch_norm to layer_norm.
    This is to preserve the privacy of the data among the batch - so it doesn't leak summary statistics.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 5),
            nn.ReLU(),  # Use Bounded Relu
            nn.Linear(5, 2),
        )

    def forward(self, x) -> torch.Tensor:
        logits = self.linear_relu_stack(x)
        return logits


def _generate_training_batch(
    batch_size: int, input_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a random input/label pair to use as input to the model.

    There are 2 classes, and a sample's class is determined by the sum of its values
    against a fixed threshold.
    """
    sample_shape = [batch_size, input_dim]
    input_sample = torch.randn(*sample_shape)
    label = (input_sample.view(batch_size, -1).sum(dim=1) > 0.325).long()
    return input_sample, label


def validate_model(model: SimpleNN) -> float:
    """
    Validate a provided model against unseen data.

    Generates a new batch of training data (32 samples), runs a forward pass of the
    server model and compares the predictions against the labels.

    Input:
        model: A SimpleNN model which will be evaluated against new data

    Returns:
        The proportion of correct predictions
    """
    new_sample, targets = _generate_training_batch(32, 8)

    result = model(new_sample)
    _, prediction = torch.max(result, dim=1)
    return sum(targets == prediction) / prediction.numel()


class Gateway:
    def __init__(
        self,
        name: str,
        batches_per_round: int = 4,
        learning_rate: float = 0.01,
        clipping_norm: float = 0.2,
        clipping_norm_type: int = 2,
        noise_multiplier: float = 0.1,
    ):
        self.model = SimpleNN()
        self.name = name
        self.batches_per_round = batches_per_round
        self.learning_rate = learning_rate
        self.clipping_norm = clipping_norm
        self.clipping_norm_type = clipping_norm_type
        self.noise_multiplier = noise_multiplier
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def add_noise(self) -> None:
        """
        Add noise to the gradients of the model's parameters.
        https://flower.ai/docs/framework/explanation-differential-privacy.html
        """
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0.0, std=self.noise_multiplier, size=param.grad.shape
                )
                param.grad += noise

    def clip_gradients(self) -> None:
        """
        Clip the gradients of the model's parameters.

        Could have used adaptive clipping, but for simplicity, we'll use a fixed norm.
        """
        nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.clipping_norm,
            norm_type=2,  # L2 norm clipping
        )

    def apply_dp(self) -> None:
        """
        Apply Differential Privacy (DP) by clipping the gradient norms
        then adding noise to the gradients.

        Clip first, then add noise to the gradients.

        Adapted from the following Flower documentation:
        https://flower.ai/docs/framework/explanation-differential-privacy.html
        """

        self.clip_gradients()
        self.add_noise()

    def fed_round(self) -> None:
        """
        Perform a single federation round of training using randomly generated data.

        This function should run a forward pass through the SimpleNN model, calculate the
        loss and update the optimizer.

        DP-SGD’s requirement of per-example gradient clipping is computationally expensive
        and slows down convergence.

        If more privacy is needed then we can apply DP at the Input Level. e.g.
        - dχ privacy, which is a stronger privacy guarantee than ε-differential privacy.
        It applies carefully calibrated noise to vector representation of words in a high dimension space
        as defined by word embedding models
        """
        self.model.train()
        for _ in range(self.batches_per_round):
            batch, targets = _generate_training_batch(8, 8)
            prediction = self.model(batch)
            loss = self.loss_fn(prediction, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.apply_dp()
            self.optimizer.step()

    def _update_model(self, state_dict: dict) -> None:
        """
        Update the gateway's local model with a new set of weights
        This should only be called by the Orchestrator to set the gateway's model weights
        with the newly aggregated server weights.
        """
        self.model.load_state_dict(state_dict)


class Orchestrator:
    def __init__(self, gateways: List[Gateway]):
        self.gateways = gateways
        self._server_model = None

    def train_fed_avg(self, num_rounds: int) -> SimpleNN:
        """
        Perform a federated training run for `num_rounds` rounds.

        Input:
            num_rounds: The number of training rounds to perform

        Returns:
            An aggregated model
        """
        # Initialise the server model at the start of training
        self._server_model = SimpleNN()

        for a_round in range(num_rounds):
            # Send the server model to the gateways at the start of each round
            self._update_gateways(self._server_model.state_dict())
            for gw in self.gateways:
                gw.fed_round()
            averaged_weights_state_dict = self._aggregate()
            # Update the server model with the aggregated weights
            self._server_model.load_state_dict(averaged_weights_state_dict)
        return self._server_model

    def _aggregate(self) -> dict:
        """
        Iterates through all gateways connected to the orchestrator, reading each model's
        state_dict, and taking the average of the weights across the models.

        e.g.:

        state_dict = {
            "fc1.weight': torch.Tensor(...),
            "fc1.bias': torch.Tensor(...),
            "fc2.weight': torch.Tensor(...),
            "fc2.bias': torch.Tensor(...)
        }

        Returns:
            A state dict containing the averaged weights
        """
        with torch.no_grad():
            averaged_weights = {}
            # Sum the weights of all gateways first then divide
            # by the number of gateways
            for gw in self.gateways:
                for key, value in gw.model.state_dict().items():
                    if key not in averaged_weights:
                        averaged_weights[key] = value.clone()
                    else:
                        averaged_weights[key] += value
            for key in averaged_weights:
                averaged_weights[key] /= len(self.gateways)
        return averaged_weights

    def _update_gateways(self, state_dict: dict) -> None:
        """
        Update all gateways' models with the provided state dict
        """
        for gw in self.gateways:
            gw._update_model(state_dict)


if __name__ == "__main__":

    gateways = [Gateway(f"gateway_{i}", 4) for i in range(4)]

    orch = Orchestrator(gateways)

    new_model = orch.train_fed_avg(32)

    validation_accuracy = validate_model(new_model)

    print(f"Training Complete. Validation accuracy: {validation_accuracy*100:.2f}%")
