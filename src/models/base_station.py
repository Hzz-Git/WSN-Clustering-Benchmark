"""
Base Station class for the WSN.

The base station (sink) is the data collection point outside the sensing area.
Per spec: BS at (50, 100) for a 100x100 field - at top center edge.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class BaseStation:
    """
    Represents the base station (sink) in the WSN.

    Attributes:
        x: X coordinate
        y: Y coordinate
        packets_received: Total packets received from CHs
        data_received: Total data bits received
    """
    x: float = 50.0
    y: float = 100.0
    packets_received: int = 0
    data_received: int = 0

    def distance_to(self, node_x: float, node_y: float) -> float:
        """Calculate distance from BS to a point."""
        return np.sqrt((self.x - node_x)**2 + (self.y - node_y)**2)

    def receive_data(self, bits: int):
        """Record data reception from a CH."""
        self.packets_received += 1
        self.data_received += bits

    def reset_stats(self):
        """Reset reception statistics."""
        self.packets_received = 0
        self.data_received = 0


if __name__ == "__main__":
    bs = BaseStation(x=50, y=100)
    print(f"Base Station at ({bs.x}, {bs.y})")

    # Test distance calculation
    node_pos = (25, 50)
    d = bs.distance_to(*node_pos)
    print(f"Distance to node at {node_pos}: {d:.2f} m")

    # Test data reception
    bs.receive_data(4000)
    bs.receive_data(4000)
    print(f"Packets received: {bs.packets_received}")
    print(f"Total data: {bs.data_received} bits")
