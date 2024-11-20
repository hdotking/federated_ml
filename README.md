# README
This document briefly describes the decision making process in the technical assessment.
In a production environment it would be preferred to organise these decisions as Architectural Decision Records (ADR) for the team to review before storing in an internal wiki for future reference.

Note that the default hyperparameters used in the Gateways were chosen to add noise, peform clipping and still provive a suitable accuracy for this task. For more complex learning tasks, we can vary the `noise_multiplier` and `clipping_norm` to improve fitting or regularisation.


# How to use
Install the requirements (preferably with pip in a virtual environment) by running:
```
pip install -r requirements.txt
```

Run the main task from the coding assignment from the terminal with:
```
python fed_train.py
```

Run the tests by navigating to the root directory and entering the command:
```
 pytest tests.py
 ```

## fed_train.py (The task at hand)

### Choice of DP
Clipping the gradients and adding noise within the training loop was chosen, as opposed to the weights at the end of a federated round, to enhance privacy during training. This extra security may be necessary for a NER task which is aiming to extract specific entities - while minimising the risk of leaking training data. The downsides include slower convergence the increased computational load, as the `apply_dp()` method is called multiple times per federation round.

### Setting overridable defaults for hyperparameters like `learning rate`, `clipping_norm` and `noise_multiplier`
Allowing the server to give default parameters allows the engineer to ensure consistency and strong privacy guarantees tailored to each client. All while allowing clients to adjust these parameters if need be. Allowing us to avoid the problems that arise with setting a universal learning rate and other hyperparamters for all clients.

These tradeoffs of course must be balanced with both the customers privacy requirements, model accuracy and budget for computation, etc.

### Clarify that Gateway._update_model() method is only called by the Orchestrator
A small comment for those wondering if the Gateway or Orchestrator should be calling this method.

### Federated Averaging and Telemetry
A Federated Averaging strategy was used in this task for simplicity, but if the requirements were more complex then a framework like Flower will would allow us to easily set and test different strategies to help balance performance and privacy. Logging is also greatly simplified in frameworks like Flower.

## tests.py (unit tests)

### `apply_dp()` makes two separate method calls to `add_noise()` and `clip_gradients()`
Breaking them down like this lets us test their functionality more readily in the unit tests (see `tests.py`). 

### Use of pytest fixtures
These are composable objects that do not carry state between tests. This provides much needed isolation, avoids repetition and improves maintainability.

## Zero dependencies on file system or API calls
