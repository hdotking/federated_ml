import pytest
import torch
from fed_train import Gateway, Orchestrator, SimpleNN, _generate_training_batch


@pytest.fixture
def single_gateway():
    """Fixture to initialize a Gateway instance with 1 batch per round for overfitting test."""
    return Gateway("test_gateway", batches_per_round=1)


@pytest.fixture
def multiple_gateways():
    """Fixture to initialize multiple gateways for orchestrator tests."""
    return [Gateway(f"gateway_{i}") for i in range(3)]


@pytest.fixture
def orchestrator(multiple_gateways):
    """Fixture to initialize an Orchestrator instance with multiple gateways."""
    return Orchestrator(multiple_gateways)


@pytest.fixture
def create_gateways_with_modified_weights(multiple_gateways):
    gw1, gw2, gw3 = multiple_gateways[:3]

    # Modify the weights of gw1 and gw2 to break any potential symmetry with gw3
    large_value = 10_000
    for param1, param2 in zip(gw1.model.parameters(), gw2.model.parameters()):
        with torch.no_grad():
            param1.add_(large_value)  # Add large_value to gw1's weights
            param2.sub_(large_value)  # Subtract large_value from gw2's weights
    return gw1, gw2, gw3


def test_gateway_model_can_overfit(single_gateway):
    """
    Trains the model on a single batch of data and checks if it can achieve 100% accuracy.
    This tells us whether the chosen model need to be more complex to learn the data.
    """
    gateway = single_gateway
    for _ in range(1000):
        gateway.fed_round()
    test_batch, test_targets = _generate_training_batch(8, 8)

    gateway.model.eval()
    with torch.no_grad():
        test_prediction = gateway.model(test_batch)
    test_accuracy = (test_prediction.argmax(dim=1) == test_targets).float().mean()

    # Assert that the model can overfit (achieve 100% accuracy)
    assert test_accuracy == 1.0, "Model should be able to overfit the training data"


def test_updates_gateways(orchestrator, multiple_gateways):
    """
    Test that the orchestrator correctly updates gateways with the server model's weights.
    """
    orchestrator._server_model = SimpleNN()
    server_model_state_dict = orchestrator._server_model.state_dict()
    initial_gateway_states = [gw.model.state_dict() for gw in multiple_gateways]
    # Ensure that the initial states are sufficiently different from the server model's state
    for initial_state in initial_gateway_states:
        for key in server_model_state_dict:
            assert not torch.equal(
                initial_state[key], server_model_state_dict[key]
            ), "Gateway model should not initially match server model."

    # Update gateways with the server model's weights
    orchestrator._update_gateways(server_model_state_dict)
    # Check that each gateway's model is updated correctly
    for gateway in multiple_gateways:
        updated_state = gateway.model.state_dict()
        for key in updated_state:
            assert torch.equal(
                updated_state[key], server_model_state_dict[key]
            ), "Gateway model did not receive the correct updated weights from the orchestrator."


def test_aggregation_logic(orchestrator, create_gateways_with_modified_weights):
    """
    Have three very different model weights and prove that the aggregation logic works as expected.
    """
    gw1, gw2, gw3 = create_gateways_with_modified_weights
    # Ensure that gw1 and gw2's weights are sufficiently different from gw3's
    for param1, param2 in zip(gw1.model.parameters(), gw2.model.parameters()):
        assert not torch.allclose(
            param1, param2, atol=1e4
        ), "gw1 and gw2 should have VERY different weights"

    aggregated_state_dict = orchestrator._aggregate()
    # Manually compute the expected average of gw1, gw2, and gw3's weights
    expected_averaged_state_dict = {}
    for key in gw1.model.state_dict().keys():
        expected_value = (
            gw1.model.state_dict()[key]
            + gw2.model.state_dict()[key]
            + gw3.model.state_dict()[key]
        ) / 3
        expected_averaged_state_dict[key] = expected_value
    # Check that the aggregated weights match the manually computed average
    for key in aggregated_state_dict:
        assert torch.allclose(
            aggregated_state_dict[key], expected_averaged_state_dict[key], atol=1e-6
        ), f"Mismatch in aggregated weight for {key}"
