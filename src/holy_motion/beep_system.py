#!/usr/bin/env python3
"""
Boxing Data Collection - Beep System

Generates audio beeps and voice announcements for punch timing.
Uses Windows winsound only.
"""

import time
import random
import winsound
from typing import Optional, List, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# pyttsx3 disabled - conflicts with bleak's asyncio/COM on Windows
# Voice announcements replaced with print + beeps
PYTTSX3_AVAILABLE = False


class BeepType(Enum):
    """Types of audio signals"""
    BEEP = "beep"
    COUNTDOWN = "countdown"
    START = "start"
    END = "end"
    VOICE = "voice"


@dataclass
class BeepEvent:
    """Record of a beep event"""
    beep_number: int
    timestamp_ms: float
    beep_type: BeepType
    announcement: Optional[str] = None
    interval_before_ms: Optional[float] = None


class BeepSystem:
    """
    Audio beep system for boxing data collection.
    Uses Windows winsound for all audio.
    """

    def __init__(self):
        self.beep_log: List[BeepEvent] = []
        self.session_start_time: Optional[float] = None

    def play_beep(self, frequency: int = 880, duration_ms: int = 150):
        """Play a beep using winsound"""
        winsound.Beep(frequency, duration_ms)

    def play_system_sound(self, sound_type: str = "asterisk"):
        """Play a Windows system sound"""
        sounds = {
            "asterisk": winsound.MB_ICONASTERISK,
            "exclamation": winsound.MB_ICONEXCLAMATION,
            "hand": winsound.MB_ICONHAND,
            "question": winsound.MB_ICONQUESTION,
            "default": winsound.MB_OK,
        }
        winsound.MessageBeep(sounds.get(sound_type, winsound.MB_OK))

    def play_countdown(self, count: int = 3):
        """Play countdown beeps (3-2-1-GO)"""
        for i in range(count, 0, -1):
            print(f"  {i}...", flush=True)
            winsound.Beep(440, 100)
            time.sleep(1.0)

        print("  GO!", flush=True)
        winsound.Beep(880, 300)
        time.sleep(0.3)

    def announce(self, text: str):
        """Text announcement with distinct beep pattern"""
        text_upper = text.upper()
        print(f"\n  >> {text_upper} <<", flush=True)

        # Different beep patterns for different punch/movement types
        if "JAB" in text_upper:
            # JAB: Single high-pitched beep (straight punch)
            winsound.Beep(1000, 250)
        elif "HOOK" in text_upper:
            # HOOK: Rising sweep (low to high) - like the arc of a hook
            winsound.Beep(400, 150)
            winsound.Beep(700, 150)
            winsound.Beep(900, 200)
        else:
            # HARD NEGATIVE: Quick low beep (do any punch-like movement that's NOT a punch)
            winsound.Beep(300, 150)

        time.sleep(0.15)

    def signal_punch(
        self,
        beep_number: int,
        announcement: Optional[str] = None,
        interval_before_ms: Optional[float] = None
    ) -> BeepEvent:
        """Signal for a punch with optional voice announcement."""
        timestamp = datetime.now().timestamp() * 1000

        if announcement:
            self.announce(announcement)
            time.sleep(0.3)

        winsound.Beep(880, 150)

        event = BeepEvent(
            beep_number=beep_number,
            timestamp_ms=timestamp,
            beep_type=BeepType.VOICE if announcement else BeepType.BEEP,
            announcement=announcement,
            interval_before_ms=interval_before_ms
        )
        self.beep_log.append(event)

        return event

    def start_session(self):
        """Mark session start with rising fanfare"""
        self.beep_log = []
        self.session_start_time = datetime.now().timestamp() * 1000

        print("\n  >> SESSION STARTING <<", flush=True)
        # Rising fanfare: low to high
        winsound.Beep(400, 150)
        winsound.Beep(500, 150)
        winsound.Beep(600, 150)
        winsound.Beep(800, 300)
        time.sleep(0.5)
        self.play_countdown(3)

    def end_session(self):
        """Mark session end with victory fanfare"""
        print("\n  >> SESSION COMPLETE <<", flush=True)
        # Victory fanfare: triumphant pattern
        winsound.Beep(600, 150)
        winsound.Beep(600, 150)
        winsound.Beep(600, 150)
        winsound.Beep(800, 400)
        time.sleep(0.1)
        winsound.Beep(700, 150)
        winsound.Beep(800, 500)

    def run_beep_sequence(
        self,
        num_beeps: int,
        interval_range: Tuple[float, float],
        announcements: Optional[List[str]] = None,
        callback: Optional[Callable[[BeepEvent], None]] = None
    ) -> List[BeepEvent]:
        """Run a sequence of beeps with randomized intervals."""
        min_interval, max_interval = interval_range
        last_time = datetime.now().timestamp() * 1000

        for i in range(num_beeps):
            interval_sec = random.uniform(min_interval, max_interval)
            time.sleep(interval_sec)

            announcement = None
            if announcements and i < len(announcements):
                announcement = announcements[i]

            current_time = datetime.now().timestamp() * 1000
            interval_ms = current_time - last_time
            last_time = current_time

            event = self.signal_punch(
                beep_number=i + 1,
                announcement=announcement,
                interval_before_ms=interval_ms
            )

            if callback:
                callback(event)

            print(f"  [{i+1}/{num_beeps}]", flush=True)

        return self.beep_log

    def get_beep_log(self) -> List[BeepEvent]:
        """Get the beep event log"""
        return self.beep_log

    def cleanup(self):
        """Cleanup (no-op for winsound)"""
        pass


def generate_punch_sequence(num_jabs: int, num_hooks: int, randomize: bool = True) -> List[str]:
    """Generate a sequence of punch announcements."""
    sequence = ["JAB"] * num_jabs + ["HOOK"] * num_hooks
    if randomize:
        random.shuffle(sequence)
    return sequence


def generate_hard_negative_sequence(num_movements: int = 20) -> List[str]:
    """Generate a sequence of hard negative movement announcements."""
    movements = ["REACH", "SWAT", "FACE", "WIPE", "POINT", "PARRY", "DRINK"]
    return [random.choice(movements) for _ in range(num_movements)]


if __name__ == "__main__":
    print("Testing Beep System")
    print("="*50)

    beep = BeepSystem()

    print("\nTesting single beep...")
    beep.play_beep()
    time.sleep(1)

    print("\nTesting countdown...")
    beep.play_countdown(3)

    print("\nTesting punch signals...")
    beep.signal_punch(1, announcement="JAB")
    time.sleep(2)
    beep.signal_punch(2, announcement="HOOK")

    print("\nDone!")
