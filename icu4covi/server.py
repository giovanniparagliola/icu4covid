import flwr as fl
from flwr.common import Metrics
from typing import List, Tuple

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,  # Sample 10% of available clients for the next round
    min_fit_clients=3,  # Minimum number of clients to be sampled for the next round
    min_available_clients=3,  # Minimum number of clients that need to be connected to the server before a training round can start,
    evaluate_metrics_aggregation_fn =weighted_average,
    fit_metrics_aggregation_fn =weighted_average
)
fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3), server_address="172.16.1.21:8080", strategy=strategy)