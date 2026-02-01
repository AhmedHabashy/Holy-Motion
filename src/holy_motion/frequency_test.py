#!/usr/bin/env python3
"""
Holyiot-23033 Frequency Test Script

This script tests and verifies the actual data frequency received from
the sensor compared to the requested frequency.

It measures:
- Actual packet rate (Hz)
- Expected vs actual frequency comparison
- Packet timing jitter
- Data loss estimation
"""

import asyncio
import argparse
import sys
import statistics
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, field

try:
    from bleak import BleakClient, BleakScanner
    from bleak.backends.characteristic import BleakGATTCharacteristic
except ImportError:
    print("Error: bleak library not installed.")
    print("Please install it with: pip install bleak")
    sys.exit(1)


# UUIDs for Holy-Motion Protocol
SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
WRITE_UUID = "6E400010-B5A3-F393-E0A9-E50E24DCCA9E"
NOTIFY_UUID = "6E400011-B5A3-F393-E0A9-E50E24DCCA9E"

DEVICE_NAME = "Holy-Motion"

# Packet types
PACKET_ACCEL = 0x21
PACKET_GYRO = 0x22
PACKET_MAGN = 0x23


@dataclass
class FrequencyStats:
    """Statistics for frequency analysis"""
    packet_type: str
    total_packets: int = 0
    expected_hz: float = 0.0
    actual_hz: float = 0.0
    min_interval_ms: float = 0.0
    max_interval_ms: float = 0.0
    avg_interval_ms: float = 0.0
    std_dev_ms: float = 0.0
    jitter_ms: float = 0.0
    accuracy_percent: float = 0.0
    timestamps: List[float] = field(default_factory=list)


class FrequencyTester:
    """Test and analyze sensor data frequency"""

    def __init__(self, device_address: Optional[str] = None):
        self.device_address = device_address
        self.client: Optional[BleakClient] = None
        self.connected = False
        self.test_running = False

        # Timing data
        self.packet_timestamps: Dict[int, List[float]] = {
            PACKET_ACCEL: [],
            PACKET_GYRO: [],
            PACKET_MAGN: [],
        }

        self.start_time: Optional[float] = None
        self.expected_interval_ms: int = 25  # ~40 Hz actual

    @staticmethod
    def calculate_checksum(data: bytes) -> int:
        """Calculate checksum"""
        return sum(data) & 0xFF

    async def scan_for_device(self, timeout: float = 10.0) -> Optional[str]:
        """Scan for device"""
        print(f"Scanning for '{DEVICE_NAME}' device...")
        devices = await BleakScanner.discover(timeout=timeout)

        for device in devices:
            if device.name and DEVICE_NAME in device.name:
                print(f"Found: {device.name} [{device.address}]")
                return device.address

        print("Device not found. Available devices:")
        for device in devices:
            if device.name:
                print(f"  {device.name} [{device.address}]")
        return None

    async def connect(self) -> bool:
        """Connect to sensor"""
        if not self.device_address:
            self.device_address = await self.scan_for_device()

        if not self.device_address:
            return False

        print(f"Connecting to {self.device_address}...")

        try:
            self.client = BleakClient(self.device_address)
            await self.client.connect()
            self.connected = True
            print("Connected!")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from sensor"""
        if self.client and self.connected:
            await self._disable_sensors()
            await self.client.disconnect()
            self.connected = False
            print("Disconnected")

    def _notification_handler(self, characteristic: BleakGATTCharacteristic, data: bytearray):
        """Record packet timestamps"""
        if not self.test_running:
            return

        timestamp = datetime.now().timestamp()
        data_bytes = bytes(data)

        if len(data_bytes) >= 2 and data_bytes[0] == 0xF6:
            packet_type = data_bytes[1]
            if packet_type in self.packet_timestamps:
                self.packet_timestamps[packet_type].append(timestamp)

    async def _send_command(self, cmd: int, data: bytes = b''):
        """Send command to sensor"""
        if not self.client:
            return

        length = len(data) + 1
        packet = bytes([0xF6, cmd, 0xF6, length]) + data
        checksum = self.calculate_checksum(packet)
        packet = packet + bytes([checksum])

        await self.client.write_gatt_char(WRITE_UUID, packet)
        await asyncio.sleep(0.1)  # Wait for response

    async def _set_interval(self, interval_ms: int):
        """Set data interval"""
        interval_value = interval_ms // 2
        await self._send_command(0x01, bytes([interval_value]))

    async def _enable_sensors(self, accel: bool = True, gyro: bool = True, magn: bool = False):
        """Enable sensors"""
        data = bytes([
            1 if accel else 0,
            1 if gyro else 0,
            1 if magn else 0,
            0,  # quaternion
            0   # euler
        ])
        await self._send_command(0x05, data)

    async def _disable_sensors(self):
        """Disable all sensors"""
        await self._send_command(0x05, bytes([0, 0, 0, 0, 0]))

    def _calculate_stats(self, packet_type: int, name: str) -> FrequencyStats:
        """Calculate frequency statistics for a packet type"""
        timestamps = self.packet_timestamps[packet_type]
        stats = FrequencyStats(packet_type=name)
        stats.total_packets = len(timestamps)
        stats.expected_hz = 1000.0 / self.expected_interval_ms

        if len(timestamps) < 2:
            return stats

        # Calculate intervals between packets
        intervals = []
        for i in range(1, len(timestamps)):
            interval_ms = (timestamps[i] - timestamps[i-1]) * 1000
            intervals.append(interval_ms)

        if not intervals:
            return stats

        # Calculate statistics
        total_duration = timestamps[-1] - timestamps[0]
        stats.actual_hz = (len(timestamps) - 1) / total_duration if total_duration > 0 else 0

        stats.min_interval_ms = min(intervals)
        stats.max_interval_ms = max(intervals)
        stats.avg_interval_ms = statistics.mean(intervals)

        if len(intervals) > 1:
            stats.std_dev_ms = statistics.stdev(intervals)
        else:
            stats.std_dev_ms = 0.0

        stats.jitter_ms = stats.max_interval_ms - stats.min_interval_ms

        if stats.expected_hz > 0:
            stats.accuracy_percent = (stats.actual_hz / stats.expected_hz) * 100

        stats.timestamps = timestamps
        return stats

    def _print_stats(self, stats: FrequencyStats):
        """Print statistics for a sensor"""
        print(f"\n{'='*60}")
        print(f"  {stats.packet_type} Statistics")
        print(f"{'='*60}")
        print(f"  Total Packets:      {stats.total_packets}")
        print(f"  Expected Frequency: {stats.expected_hz:.1f} Hz")
        print(f"  Actual Frequency:   {stats.actual_hz:.2f} Hz")
        print(f"  Accuracy:           {stats.accuracy_percent:.1f}%")
        print(f"  ")
        print(f"  Interval (expected): {self.expected_interval_ms:.1f} ms")
        print(f"  Interval (actual):   {stats.avg_interval_ms:.2f} ms")
        print(f"  Interval (min):      {stats.min_interval_ms:.2f} ms")
        print(f"  Interval (max):      {stats.max_interval_ms:.2f} ms")
        print(f"  Std Deviation:       {stats.std_dev_ms:.2f} ms")
        print(f"  Jitter (max-min):    {stats.jitter_ms:.2f} ms")

    async def run_test(self, interval_ms: int, duration_sec: int, include_magn: bool = False):
        """Run frequency test"""
        self.expected_interval_ms = interval_ms

        # Clear previous data
        for key in self.packet_timestamps:
            self.packet_timestamps[key] = []

        print(f"\n{'#'*60}")
        print(f"  FREQUENCY TEST")
        print(f"{'#'*60}")
        print(f"  Requested interval: {interval_ms} ms")
        print(f"  Expected frequency: {1000/interval_ms:.1f} Hz")
        print(f"  Test duration:      {duration_sec} seconds")
        print(f"  Sensors:            Accelerometer, Gyroscope" + (", Magnetometer" if include_magn else ""))
        print(f"{'#'*60}\n")

        # Start notifications
        await self.client.start_notify(NOTIFY_UUID, self._notification_handler)

        # Configure sensor
        print("Configuring sensor...")
        await self._set_interval(interval_ms)
        await asyncio.sleep(0.2)

        await self._enable_sensors(accel=True, gyro=True, magn=include_magn)
        await asyncio.sleep(0.5)

        # Start test
        print(f"Collecting data for {duration_sec} seconds...")
        print("Progress: ", end="", flush=True)

        self.test_running = True
        self.start_time = datetime.now().timestamp()

        # Show progress
        for i in range(duration_sec):
            await asyncio.sleep(1)
            print(".", end="", flush=True)

        self.test_running = False
        print(" Done!\n")

        # Stop sensor
        await self._disable_sensors()
        await self.client.stop_notify(NOTIFY_UUID)

        # Calculate and display results
        print("\n" + "="*60)
        print("  TEST RESULTS")
        print("="*60)

        accel_stats = self._calculate_stats(PACKET_ACCEL, "ACCELEROMETER")
        gyro_stats = self._calculate_stats(PACKET_GYRO, "GYROSCOPE")

        self._print_stats(accel_stats)
        self._print_stats(gyro_stats)

        if include_magn:
            magn_stats = self._calculate_stats(PACKET_MAGN, "MAGNETOMETER")
            self._print_stats(magn_stats)

        # Summary
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")

        all_accuracies = [accel_stats.accuracy_percent, gyro_stats.accuracy_percent]
        avg_accuracy = statistics.mean([a for a in all_accuracies if a > 0]) if any(a > 0 for a in all_accuracies) else 0

        print(f"  Average Accuracy: {avg_accuracy:.1f}%")

        if avg_accuracy >= 95:
            print(f"  Status: EXCELLENT - Frequency is very accurate")
        elif avg_accuracy >= 85:
            print(f"  Status: GOOD - Frequency is within acceptable range")
        elif avg_accuracy >= 70:
            print(f"  Status: FAIR - Some packet loss or timing variation")
        else:
            print(f"  Status: POOR - Significant frequency deviation")

        print(f"\n  Total packets received:")
        print(f"    Accelerometer: {accel_stats.total_packets}")
        print(f"    Gyroscope:     {gyro_stats.total_packets}")

        expected_packets = int(duration_sec * 1000 / interval_ms)
        print(f"    Expected:      ~{expected_packets} per sensor")

        return accel_stats, gyro_stats


async def main():
    parser = argparse.ArgumentParser(
        description="Holyiot-23033 Frequency Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python frequency_test.py -i 20 -d 10    # Test ~40Hz for 10 seconds
  python frequency_test.py -i 50 -d 30    # Test ~20Hz for 30 seconds

Note: With only accel+gyro enabled, actual rate is ~40 Hz (25ms median interval).
        """
    )

    parser.add_argument(
        "-a", "--address",
        type=str,
        default="C9:3B:9C:CF:58:E8",
        help="Bluetooth address (default: C9:3B:9C:CF:58:E8)"
    )

    parser.add_argument(
        "-i", "--interval",
        type=int,
        default=20,
        help="Data interval in ms (default: 20ms). Actual rate ~40 Hz."
    )

    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=10,
        help="Test duration in seconds (default: 10)"
    )

    parser.add_argument(
        "--magn",
        action="store_true",
        help="Also test magnetometer"
    )

    args = parser.parse_args()

    tester = FrequencyTester(device_address=args.address)

    try:
        if not await tester.connect():
            print("Failed to connect")
            return

        await tester.run_test(
            interval_ms=args.interval,
            duration_sec=args.duration,
            include_magn=args.magn
        )

    except KeyboardInterrupt:
        print("\n\nTest interrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await tester.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
