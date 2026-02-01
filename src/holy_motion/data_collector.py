#!/usr/bin/env python3
"""
Boxing Data Collection - Main Script

This is the main entry point for collecting boxing punch data
using the Holyiot-23033 sensor.

Usage:
    python data_collector.py                  # Interactive menu
    python data_collector.py --day 2          # Run specific day
    python data_collector.py --session 4      # Run specific session
    python data_collector.py --list           # List all sessions
"""

import asyncio
import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from session_configs import (
    ALL_SESSIONS, SESSIONS_BY_DAY, SESSIONS_BY_ID, DAY_NAMES, DAY_FOLDERS,
    SessionConfig, SessionType, PunchType, COMMON_CHECKLIST,
    get_session, get_day_sessions, print_session_summary
)
from beep_system import BeepSystem, BeepEvent, generate_punch_sequence, generate_hard_negative_sequence

try:
    from bleak import BleakClient, BleakScanner
    from bleak.backends.characteristic import BleakGATTCharacteristic
except ImportError:
    print("Error: bleak library not installed.")
    print("Please install with: pip install bleak")
    sys.exit(1)


# Sensor configuration
SENSOR_ADDRESS = "C9:3B:9C:CF:58:E8"
SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
WRITE_UUID = "6E400010-B5A3-F393-E0A9-E50E24DCCA9E"
NOTIFY_UUID = "6E400011-B5A3-F393-E0A9-E50E24DCCA9E"

# Data packet types
PACKET_ACCEL = 0x21
PACKET_GYRO = 0x22
PACKET_MAGN = 0x23
PACKET_QUATERNION = 0x25
PACKET_EULER = 0x26

# Base directory for data
BASE_DIR = Path(__file__).parent / "days"


@dataclass
class SensorReading:
    """Single sensor reading with all data"""
    timestamp_ms: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    euler_roll: float
    euler_pitch: float
    euler_yaw: float
    quat_w: float
    quat_x: float
    quat_y: float
    quat_z: float


class BoxingDataCollector:
    """
    Main data collector for boxing sessions.

    Handles sensor connection, data collection, and file saving.
    """

    def __init__(self, sensor_address: str = SENSOR_ADDRESS):
        self.sensor_address = sensor_address
        self.client: Optional[BleakClient] = None
        self.connected = False
        self.collecting = False

        # Data storage
        self.readings: List[SensorReading] = []
        self.last_gyro = (0.0, 0.0, 0.0)
        self.last_euler = (0.0, 0.0, 0.0)
        self.last_quat = (0.0, 0.0, 0.0, 0.0)

        # Beep system
        self.beep_system = BeepSystem()

        # Current session
        self.current_session: Optional[SessionConfig] = None

    @staticmethod
    def calculate_checksum(data: bytes) -> int:
        """Calculate checksum for command"""
        return sum(data) & 0xFF

    async def connect(self) -> bool:
        """Connect to sensor"""
        print(f"\nConnecting to sensor {self.sensor_address}...")

        try:
            self.client = BleakClient(self.sensor_address)
            await self.client.connect()
            self.connected = True
            print("Connected successfully!")
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
            print("Disconnected from sensor")

    async def _send_command(self, cmd: int, data: bytes = b''):
        """Send command to sensor"""
        if not self.client:
            return

        length = len(data) + 1
        packet = bytes([0xF6, cmd, 0xF6, length]) + data
        checksum = self.calculate_checksum(packet)
        packet = packet + bytes([checksum])

        await self.client.write_gatt_char(WRITE_UUID, packet)
        await asyncio.sleep(0.1)

    async def _enable_sensors(self):
        """Enable sensors for data collection"""
        # Set interval to 20ms (10 * 2ms) - request ~50Hz, expect ~40Hz actual
        await self._send_command(0x01, bytes([10]))

        # Enable ONLY accel + gyro (fewer sensors = less BLE traffic = better rate)
        # Format: [accel, gyro, magn, quaternion, euler]
        await self._send_command(0x05, bytes([1, 1, 0, 0, 0]))

    async def _disable_sensors(self):
        """Disable all sensors"""
        await self._send_command(0x05, bytes([0, 0, 0, 0, 0]))

    def _parse_packet(self, data: bytes):
        """Parse incoming sensor packet"""
        if len(data) < 4 or data[0] != 0xF6 or data[2] != 0xF6:
            return

        packet_type = data[1]
        timestamp = datetime.now().timestamp() * 1000

        import struct

        # Save on every ACCEL packet (most reliable trigger)
        if packet_type == PACKET_ACCEL and len(data) >= 10:
            x = struct.unpack('>h', data[4:6])[0] * 0.001
            y = struct.unpack('>h', data[6:8])[0] * 0.001
            z = struct.unpack('>h', data[8:10])[0] * 0.001

            if self.collecting:
                reading = SensorReading(
                    timestamp_ms=timestamp,
                    accel_x=x,
                    accel_y=y,
                    accel_z=z,
                    gyro_x=self.last_gyro[0],
                    gyro_y=self.last_gyro[1],
                    gyro_z=self.last_gyro[2],
                    euler_roll=self.last_euler[0],
                    euler_pitch=self.last_euler[1],
                    euler_yaw=self.last_euler[2],
                    quat_w=self.last_quat[0],
                    quat_x=self.last_quat[1],
                    quat_y=self.last_quat[2],
                    quat_z=self.last_quat[3],
                )
                self.readings.append(reading)

        elif packet_type == PACKET_GYRO and len(data) >= 10:
            x = struct.unpack('>h', data[4:6])[0] * 0.1
            y = struct.unpack('>h', data[6:8])[0] * 0.1
            z = struct.unpack('>h', data[8:10])[0] * 0.1
            self.last_gyro = (x, y, z)

        elif packet_type == PACKET_EULER and len(data) >= 10:
            roll = struct.unpack('>h', data[4:6])[0] * 0.01
            pitch = struct.unpack('>h', data[6:8])[0] * 0.01
            yaw = struct.unpack('>h', data[8:10])[0] * 0.01
            self.last_euler = (roll, pitch, yaw)

        elif packet_type == PACKET_QUATERNION and len(data) >= 12:
            w = struct.unpack('>h', data[4:6])[0] * 0.001
            x = struct.unpack('>h', data[6:8])[0] * 0.001
            y = struct.unpack('>h', data[8:10])[0] * 0.001
            z = struct.unpack('>h', data[10:12])[0] * 0.001
            self.last_quat = (w, x, y, z)

    def _notification_handler(self, characteristic: BleakGATTCharacteristic, data: bytearray):
        """Handle incoming notifications"""
        self._parse_packet(bytes(data))

    async def start_collection(self):
        """Start data collection"""
        self.readings = []
        self.last_gyro = (0.0, 0.0, 0.0)
        self.last_euler = (0.0, 0.0, 0.0)
        self.last_quat = (0.0, 0.0, 0.0, 0.0)
        self.collecting = True

        await self.client.start_notify(NOTIFY_UUID, self._notification_handler)
        await self._enable_sensors()

    async def stop_collection(self):
        """Stop data collection"""
        self.collecting = False
        await self._disable_sensors()
        await self.client.stop_notify(NOTIFY_UUID)

    def save_data(self, session: SessionConfig, output_dir: Path) -> tuple:
        """
        Save collected data to CSV files.

        Returns:
            Tuple of (data_file_path, beep_file_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        punch_type = session.punch_type.value

        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save sensor data
        data_filename = f"session_{session.session_id:02d}_{punch_type}_{timestamp}_data.csv"
        data_path = output_dir / data_filename

        with open(data_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp_ms', 'accel_x', 'accel_y', 'accel_z',
                'gyro_x', 'gyro_y', 'gyro_z',
                'euler_roll', 'euler_pitch', 'euler_yaw',
                'quat_w', 'quat_x', 'quat_y', 'quat_z'
            ])
            for reading in self.readings:
                writer.writerow([
                    reading.timestamp_ms,
                    reading.accel_x, reading.accel_y, reading.accel_z,
                    reading.gyro_x, reading.gyro_y, reading.gyro_z,
                    reading.euler_roll, reading.euler_pitch, reading.euler_yaw,
                    reading.quat_w, reading.quat_x, reading.quat_y, reading.quat_z
                ])

        # Save beep log
        beep_filename = f"session_{session.session_id:02d}_{punch_type}_{timestamp}_beeps.csv"
        beep_path = output_dir / beep_filename

        with open(beep_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['beep_number', 'timestamp_ms', 'beep_type', 'announcement', 'interval_before_ms'])
            for event in self.beep_system.get_beep_log():
                writer.writerow([
                    event.beep_number,
                    event.timestamp_ms,
                    event.beep_type.value,
                    event.announcement or '',
                    event.interval_before_ms or ''
                ])

        return data_path, beep_path

    def display_session_info(self, session: SessionConfig):
        """Display session information and instructions"""
        print("\n" + "=" * 70)
        print(f"  DAY {session.day} - SESSION {session.session_id}: {session.name.upper()}")
        print("=" * 70)
        print(f"  Type: {session.session_type.value} ({session.session_type.name})")
        print(f"  Punch: {session.punch_type.value}")
        print(f"  Intensity: {session.intensity.value}")
        print(f"  Speed: {session.speed.value}")
        print(f"  Number of punches: {session.num_punches}")
        if session.interval_range[0] > 0:
            print(f"  Interval: {session.interval_range[0]}-{session.interval_range[1]} seconds")
        print(f"  Estimated duration: {session.duration_estimate_sec // 60}:{session.duration_estimate_sec % 60:02d}")
        print()
        print("  INSTRUCTIONS:")
        for line in session.instructions:
            print(f"  {line}")
        print()
        if session.notes:
            print(f"  NOTE: {session.notes}")
            print()
        print("=" * 70)

    def display_checklist(self, session: SessionConfig):
        """Display pre-session checklist"""
        checklist = session.pre_checklist if session.pre_checklist else COMMON_CHECKLIST

        print("\n  PRE-SESSION CHECKLIST:")
        print("  " + "-" * 40)
        for item in checklist:
            print(f"  [ ] {item}")
        print()

    async def run_session(self, session: SessionConfig) -> bool:
        """
        Run a complete data collection session.

        Returns:
            True if session completed successfully
        """
        self.current_session = session

        # Display session info
        self.display_session_info(session)
        self.display_checklist(session)

        # Wait for user confirmation
        input("\n  Press ENTER when ready to begin...")

        # Connect to sensor
        if not self.connected:
            if not await self.connect():
                print("  ERROR: Could not connect to sensor")
                return False

        # Start data collection
        print("\n  Starting data collection...")
        await self.start_collection()

        # Generate announcements based on session type and punch type
        announcements = None
        if session.session_type == SessionType.SPECIAL:
            # Edge cases - no announcements, just beeps (user does random edge case)
            announcements = None
        elif session.session_type == SessionType.D_HARD_NEGATIVE:
            # Hard negatives - no announcements, just beeps (user does random non-punch)
            announcements = None
        elif session.session_type == SessionType.C_MIXED or session.punch_type == PunchType.MIXED:
            # Mixed punches - generate randomized jab/hook sequence
            jabs = session.num_punches // 2
            hooks = session.num_punches - jabs
            announcements = generate_punch_sequence(jabs, hooks)
        elif session.punch_type == PunchType.JAB:
            announcements = ["JAB"] * session.num_punches
        elif session.punch_type == PunchType.HOOK:
            announcements = ["HOOK"] * session.num_punches

        # Run beep sequence in separate thread to avoid blocking BLE notifications
        # Also check for D_HARD_NEGATIVE which has num_punches=0 but still needs beep sequence
        if session.num_punches > 0 or session.session_type == SessionType.D_HARD_NEGATIVE:
            def run_beeps():
                self.beep_system.start_session()
                self.beep_system.run_beep_sequence(
                    num_beeps=session.num_punches if session.session_type != SessionType.D_HARD_NEGATIVE else 20,
                    interval_range=session.interval_range,
                    announcements=announcements
                )
                self.beep_system.end_session()

            # Run beeps in thread so time.sleep() doesn't block asyncio event loop
            await asyncio.to_thread(run_beeps)
        else:
            # Idle session - just collect for duration
            self.beep_system.announce("Begin idle recording")
            print(f"\n  Recording for {session.duration_estimate_sec} seconds...")
            await asyncio.sleep(session.duration_estimate_sec)
            self.beep_system.announce("Recording complete")

        # Stop data collection
        await self.stop_collection()

        # Get output directory
        day_folder = DAY_FOLDERS[session.day]
        output_dir = BASE_DIR / day_folder / session.folder_name

        # Save data
        data_path, beep_path = self.save_data(session, output_dir)

        # Print summary
        print("\n" + "=" * 70)
        print("  SESSION COMPLETE")
        print("=" * 70)
        print(f"  Data points collected: {len(self.readings)}")
        print(f"  Beep events logged: {len(self.beep_system.get_beep_log())}")
        print(f"  Data saved to: {data_path}")
        print(f"  Beeps saved to: {beep_path}")

        # Quality check
        duration = (self.readings[-1].timestamp_ms - self.readings[0].timestamp_ms) / 1000 if self.readings else 0
        rate = len(self.readings) / duration if duration > 0 else 0
        print(f"\n  Quality Metrics:")
        print(f"    Duration: {duration:.1f} seconds")
        print(f"    Sample rate: {rate:.1f} Hz")
        print(f"    Expected ~20 Hz: {'OK' if 15 < rate < 30 else 'CHECK'}")
        print("=" * 70)

        return True

    def cleanup(self):
        """Clean up resources"""
        self.beep_system.cleanup()


def display_menu():
    """Display interactive menu"""
    print("\n" + "=" * 70)
    print("  BOXING DATA COLLECTION SYSTEM")
    print("=" * 70)
    print()
    print("  Options:")
    print("    1. Run specific session")
    print("    2. Run all sessions for a day")
    print("    3. List all sessions")
    print("    4. View session details")
    print("    5. Exit")
    print()


async def interactive_mode():
    """Run in interactive mode"""
    collector = BoxingDataCollector()

    try:
        while True:
            display_menu()
            choice = input("  Enter choice (1-5): ").strip()

            if choice == "1":
                session_id = input("  Enter session ID (0-23): ").strip()
                try:
                    session = get_session(int(session_id))
                    if session:
                        await collector.run_session(session)
                    else:
                        print(f"  Session {session_id} not found")
                except ValueError:
                    print("  Invalid session ID")

            elif choice == "2":
                day = input("  Enter day (1-7): ").strip()
                try:
                    sessions = get_day_sessions(int(day))
                    if sessions:
                        print(f"\n  Running {len(sessions)} sessions for Day {day}")
                        for session in sessions:
                            if not await collector.run_session(session):
                                break
                            cont = input("\n  Continue to next session? (y/n): ").strip().lower()
                            if cont != 'y':
                                break
                    else:
                        print(f"  Day {day} not found")
                except ValueError:
                    print("  Invalid day number")

            elif choice == "3":
                print_session_summary()

            elif choice == "4":
                session_id = input("  Enter session ID (0-23): ").strip()
                try:
                    session = get_session(int(session_id))
                    if session:
                        collector.display_session_info(session)
                    else:
                        print(f"  Session {session_id} not found")
                except ValueError:
                    print("  Invalid session ID")

            elif choice == "5":
                print("\n  Goodbye!")
                break

            else:
                print("  Invalid choice")

    finally:
        await collector.disconnect()
        collector.cleanup()


async def main():
    parser = argparse.ArgumentParser(
        description="Boxing Data Collection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_collector.py                  # Interactive menu
  python data_collector.py --day 2          # Run all Day 2 sessions
  python data_collector.py --session 4      # Run session 4 only
  python data_collector.py --list           # List all sessions
        """
    )

    parser.add_argument(
        "-d", "--day",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Run all sessions for specified day"
    )

    parser.add_argument(
        "-s", "--session",
        type=int,
        help="Run specific session by ID (0-23)"
    )

    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all sessions (0-23)"
    )

    parser.add_argument(
        "-a", "--address",
        type=str,
        default=SENSOR_ADDRESS,
        help=f"Sensor Bluetooth address (default: {SENSOR_ADDRESS})"
    )

    args = parser.parse_args()

    if args.list:
        print_session_summary()
        return

    if args.session is not None:
        session = get_session(args.session)
        if not session:
            print(f"Error: Session {args.session} not found")
            return

        collector = BoxingDataCollector(args.address)
        try:
            await collector.run_session(session)
        finally:
            await collector.disconnect()
            collector.cleanup()
        return

    if args.day:
        sessions = get_day_sessions(args.day)
        if not sessions:
            print(f"Error: Day {args.day} not found")
            return

        collector = BoxingDataCollector(args.address)
        try:
            print(f"\nRunning {len(sessions)} sessions for Day {args.day}: {DAY_NAMES[args.day]}")
            for session in sessions:
                if not await collector.run_session(session):
                    break
                if session != sessions[-1]:
                    cont = input("\n  Continue to next session? (y/n): ").strip().lower()
                    if cont != 'y':
                        break
        finally:
            await collector.disconnect()
            collector.cleanup()
        return

    # Default: interactive mode
    await interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
