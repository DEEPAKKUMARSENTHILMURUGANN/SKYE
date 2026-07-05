import unittest
import os
import torch
from federated_learning.federated import AircraftEdgeNode, FederatedAggregator

class TestFederatedLearning(unittest.TestCase):
    def setUp(self):
        self.aggregator = FederatedAggregator()
        self.node = AircraftEdgeNode(node_id=1, seed=42)

    def test_local_data_generation(self):
        self.assertIsNotNone(self.node.local_data)
        self.assertEqual(len(self.node.local_data.shape), 3)
        self.assertEqual(self.node.local_data.shape[1], 30)
        self.assertEqual(self.node.local_data.shape[2], 23)

    def test_local_training_and_aggregation(self):
        loss_before = self.aggregator.evaluate_global_model()
        global_weights = self.aggregator.global_model.state_dict()
        local_weights = self.node.train_local_model(global_weights, epochs=1)
        self.aggregator.aggregate([local_weights])
        global_after = self.aggregator.global_model.state_dict()
        for key in global_after.keys():
            self.assertTrue(torch.allclose(global_after[key], local_weights[key]))

if __name__ == "__main__":
    unittest.main()
