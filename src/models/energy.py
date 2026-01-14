"""
First-Order Radio Energy Model for WSN Simulation

Reference: Heinzelman et al., "Energy-Efficient Communication Protocol
for Wireless Microsensor Networks," HICSS 2000.

Updated for ABC protocol with spec parameters:
- E_elec = 50 nJ/bit
- E_amp = 100 pJ/bit/m^2 (simplified, single-mode)
- Also supports two-mode (free-space vs multipath)
"""

import numpy as np


class EnergyModel:
    """First-order radio energy dissipation model."""

    def __init__(
        self,
        e_elec: float = 50e-9,       # Electronics energy (J/bit)
        e_amp: float = 100e-12,      # Amplifier energy (J/bit/m^2) - from spec
        e_fs: float = 10e-12,        # Free space amplifier (J/bit/m^2)
        e_mp: float = 0.0013e-12,    # Multipath amplifier (J/bit/m^4)
        e_da: float = 5e-9,          # Data aggregation (J/bit)
        d_crossover: float = 87.0,   # Threshold distance (m)
        use_two_mode: bool = False   # If False, use simple E_amp model from spec
    ):
        self.e_elec = e_elec
        self.e_amp = e_amp  # Simplified single-mode amplifier
        self.e_fs = e_fs
        self.e_mp = e_mp
        self.e_da = e_da
        self.d_crossover = d_crossover
        self.use_two_mode = use_two_mode
    
    def tx_energy(self, bits: int, distance: float) -> float:
        """
        Energy consumed to transmit 'bits' over 'distance' meters.

        If use_two_mode=False (default, from spec):
            E_tx = E_elec * k + E_amp * k * d^2

        If use_two_mode=True (classic LEACH model):
            E_tx = E_elec * k + E_fs * k * d^2  (if d < d_crossover)
            E_tx = E_elec * k + E_mp * k * d^4  (if d >= d_crossover)
        """
        if not self.use_two_mode:
            # Simplified model from spec
            return self.e_elec * bits + self.e_amp * bits * (distance ** 2)
        else:
            # Two-mode model (free-space vs multipath)
            if distance < self.d_crossover:
                return self.e_elec * bits + self.e_fs * bits * (distance ** 2)
            else:
                return self.e_elec * bits + self.e_mp * bits * (distance ** 4)
    
    def rx_energy(self, bits: int) -> float:
        """Energy consumed to receive 'bits'."""
        return self.e_elec * bits
    
    def aggregation_energy(self, bits: int) -> float:
        """Energy consumed to aggregate 'bits' of data."""
        return self.e_da * bits
    
    def ch_energy_per_round(
        self,
        num_members: int,
        data_bits: int,
        distance_to_bs: float,
        aggregation_ratio: float = 0.5
    ) -> float:
        """
        Total energy consumed by a cluster head in one round:
        - Receive from all members
        - Aggregate all data
        - Transmit aggregated data to BS

        Args:
            num_members: Number of cluster members
            data_bits: Bits per data packet
            distance_to_bs: Distance to base station (m)
            aggregation_ratio: Ratio of output to input after aggregation
                Per spec: 0.5 (CH sends half the received data)

        Returns:
            Total energy consumed (J)
        """
        # Receive from all members
        rx = self.rx_energy(data_bits) * num_members

        # Aggregate all data (members + own)
        total_input_bits = data_bits * (num_members + 1)
        agg = self.aggregation_energy(total_input_bits)

        # Transmit aggregated data to BS
        # Per spec: aggregated output = aggregation_ratio * input
        output_bits = int(total_input_bits * aggregation_ratio)
        tx = self.tx_energy(output_bits, distance_to_bs)

        return rx + agg + tx
    
    def member_energy_per_round(
        self, 
        data_bits: int, 
        distance_to_ch: float
    ) -> float:
        """
        Total energy consumed by a cluster member in one round:
        - Sense data (negligible, not modeled)
        - Transmit to CH
        """
        return self.tx_energy(data_bits, distance_to_ch)


    def control_tx_energy(self, control_bits: int, distance: float) -> float:
        """Energy to transmit a control packet."""
        return self.tx_energy(control_bits, distance)

    def control_rx_energy(self, control_bits: int) -> float:
        """Energy to receive a control packet."""
        return self.rx_energy(control_bits)


# Vectorized versions for efficiency with numpy arrays
class VectorizedEnergyModel(EnergyModel):
    """Vectorized energy calculations for batch operations."""

    def tx_energy_batch(self, bits: int, distances: np.ndarray) -> np.ndarray:
        """Compute transmit energy for multiple distances at once."""
        if not self.use_two_mode:
            return self.e_elec * bits + self.e_amp * bits * (distances ** 2)
        else:
            return np.where(
                distances < self.d_crossover,
                self.e_elec * bits + self.e_fs * bits * (distances ** 2),
                self.e_elec * bits + self.e_mp * bits * (distances ** 4)
            )


if __name__ == "__main__":
    print("=== Energy Model Test (Spec Parameters) ===\n")

    # Default model (from spec: E_amp = 100 pJ/bit/m^2)
    model = EnergyModel()

    # Test case: 4000-bit packet over 50m
    e_tx = model.tx_energy(4000, 50)
    e_rx = model.rx_energy(4000)

    print(f"E_elec = {model.e_elec * 1e9:.0f} nJ/bit")
    print(f"E_amp = {model.e_amp * 1e12:.0f} pJ/bit/m^2")
    print(f"\nTransmit 4000 bits over 50m: {e_tx * 1e6:.4f} uJ")
    print(f"Receive 4000 bits: {e_rx * 1e6:.4f} uJ")

    # CH energy with 5 members (per spec: 5 member packets), 75m to BS
    e_ch = model.ch_energy_per_round(5, 4000 * 8, 75, aggregation_ratio=0.5)
    print(f"\nCH energy (5 members, 75m to BS, 0.5x agg): {e_ch * 1e3:.4f} mJ")

    # Member energy (transmit to CH at 20m)
    e_member = model.member_energy_per_round(4000 * 8, 20)
    print(f"Member energy (20m to CH): {e_member * 1e3:.4f} mJ")

    # Control packet energy
    e_ctrl = model.control_tx_energy(120 * 8, 30)
    print(f"\nControl packet (120B, 30m): {e_ctrl * 1e6:.4f} uJ")
