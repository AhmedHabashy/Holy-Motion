#!/usr/bin/env python3
"""
Boxing Data Collection - Session Configurations

Defines all 23 sessions across 6 days with their parameters,
instructions, and requirements.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class SessionType(Enum):
    """Session type categories"""
    A_CONTROLLED = "A"      # Single punch, controlled, minimal movement
    B_NATURAL = "B"         # Single punch, natural movement
    C_MIXED = "C"           # Mixed punches (jab + hook)
    D_HARD_NEGATIVE = "D"   # Non-punch movements
    E_IDLE = "E"            # Idle baseline, no punches
    CALIBRATION = "CAL"     # Sensor calibration
    SPECIAL = "SPECIAL"     # Edge cases, combinations


class PunchType(Enum):
    """Punch types"""
    JAB = "jab"
    HOOK = "hook"
    MIXED = "mixed"
    NONE = "none"


class Intensity(Enum):
    """Punch intensity levels"""
    LIGHT = "light"
    MEDIUM = "medium"
    POWER = "power"
    MIXED = "mixed"
    NA = "na"


class Speed(Enum):
    """Punch speed levels"""
    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"
    MIXED = "mixed"
    NA = "na"


@dataclass
class SessionConfig:
    """Configuration for a single data collection session"""
    session_id: int
    name: str
    day: int
    session_type: SessionType
    punch_type: PunchType
    intensity: Intensity
    speed: Speed
    num_punches: int
    interval_range: Tuple[float, float]  # (min_sec, max_sec) for beep intervals
    duration_estimate_sec: int
    instructions: List[str]
    pre_checklist: List[str] = field(default_factory=list)
    notes: str = ""
    folder_name: str = ""

    def __post_init__(self):
        if not self.folder_name:
            self.folder_name = f"session_{self.session_id:02d}_{self.name.lower().replace(' ', '_')}"


# Pre-session checklist (common to all sessions)
COMMON_CHECKLIST = [
    "Sensor charged (>50% battery)",
    "Sensor secured on LEFT wrist (lead hand for orthodox stance)",
    "Position matches marked location from Day 1",
    "2-minute warm-up completed",
    "Room clear of obstacles",
    "Water nearby",
]

# ============================================================================
# DAY 1: BASELINE & CALIBRATION
# ============================================================================

SESSION_00 = SessionConfig(
    session_id=0,
    name="calibration",
    day=1,
    session_type=SessionType.CALIBRATION,
    punch_type=PunchType.NONE,
    intensity=Intensity.NA,
    speed=Speed.NA,
    num_punches=0,
    interval_range=(0, 0),
    duration_estimate_sec=60,
    instructions=[
        "This session establishes sensor placement and validates connection.",
        "",
        "SENSOR PLACEMENT (Orthodox stance - LEFT wrist):",
        "  1. Place sensor flat on back of LEFT wrist (lead hand)",
        "  2. Position 2cm from wrist bone toward elbow",
        "  3. USB port should face toward elbow",
        "  4. Tighten strap - sensor should NOT move independently",
        "",
        "WHY LEFT WRIST:",
        "  - Orthodox stance: left hand throws jabs and lead hooks",
        "  - Sensor captures LEAD hand punches (jab + left hook)",
        "",
        "MARKING:",
        "  1. Use skin-safe marker to outline sensor position",
        "  2. Take photo of placement for reference",
        "",
        "VERIFICATION:",
        "  1. Shake wrist - sensor should not wobble",
        "  2. Rotate wrist - sensor should not twist",
        "  3. Shadow box for 30 seconds to verify comfort",
    ],
    notes="Photo documentation required. Mark sensor position.",
)

SESSION_01 = SessionConfig(
    session_id=1,
    name="idle",
    day=1,
    session_type=SessionType.E_IDLE,
    punch_type=PunchType.NONE,
    intensity=Intensity.NA,
    speed=Speed.NA,
    num_punches=0,
    interval_range=(0, 0),
    duration_estimate_sec=120,
    instructions=[
        "Collect baseline idle data with NO punches.",
        "",
        "ACTIVITIES (30 seconds each):",
        "  1. Stand still, guard position, breathe normally",
        "  2. Gentle weight shifting (boxer's bounce)",
        "  3. Head movement only (slips, rolls)",
        "  4. Footwork only - NO hand movement",
        "",
        "IMPORTANT: Do NOT throw any punches during this session.",
    ],
    notes="Baseline for idle class. No punches.",
)

SESSION_02 = SessionConfig(
    session_id=2,
    name="reference",
    day=1,
    session_type=SessionType.A_CONTROLLED,
    punch_type=PunchType.MIXED,
    intensity=Intensity.MEDIUM,
    speed=Speed.SLOW,
    num_punches=20,
    interval_range=(4.0, 6.0),
    duration_estimate_sec=150,
    instructions=[
        "Reference punches to verify labeling system.",
        "",
        "You will throw 10 JABS and 10 LEFT HOOKS (alternating).",
        "Voice will announce which punch before each beep.",
        "",
        "IMPORTANT: All punches are with LEFT hand (sensor hand).",
        "  - JAB = straight left punch",
        "  - HOOK = left hook (lead hook)",
        "",
        "EXECUTION:",
        "  1. Stand in orthodox stance, hands up in guard",
        "  2. Listen for voice announcement (JAB or HOOK)",
        "  3. On BEEP - throw ONE slow, controlled LEFT-hand punch",
        "  4. FULL EXTENSION - don't cut the punch short",
        "  5. Return to guard position",
        "  6. Stay still until next announcement",
        "",
        "Focus on CLEAN technique, not speed or power.",
    ],
    notes="Verify peak detection and labeling accuracy.",
)

# ============================================================================
# DAY 2: JABS
# ============================================================================

SESSION_03 = SessionConfig(
    session_id=3,
    name="jab_light_slow",
    day=2,
    session_type=SessionType.A_CONTROLLED,
    punch_type=PunchType.JAB,
    intensity=Intensity.LIGHT,
    speed=Speed.SLOW,
    num_punches=20,
    interval_range=(3.0, 6.0),
    duration_estimate_sec=120,
    instructions=[
        "Light, slow jabs - focus on technique.",
        "",
        "ON EACH BEEP:",
        "  1. Throw ONE light jab (setup/range-finder power)",
        "  2. Slow, controlled extension",
        "  3. Full arm extension - touch the target",
        "  4. Return to guard",
        "",
        "Stand relatively still between punches.",
        "Minimal footwork or head movement.",
    ],
)

SESSION_04 = SessionConfig(
    session_id=4,
    name="jab_medium",
    day=2,
    session_type=SessionType.A_CONTROLLED,
    punch_type=PunchType.JAB,
    intensity=Intensity.MEDIUM,
    speed=Speed.MEDIUM,
    num_punches=20,
    interval_range=(3.0, 6.0),
    duration_estimate_sec=120,
    instructions=[
        "Medium intensity jabs - normal sparring power.",
        "",
        "ON EACH BEEP:",
        "  1. Throw ONE medium-power jab",
        "  2. Normal speed - not rushed, not slow",
        "  3. Snap the punch - quick extension and return",
        "  4. Return to guard",
        "",
        "Stand relatively still between punches.",
    ],
)

SESSION_05 = SessionConfig(
    session_id=5,
    name="jab_power_fast",
    day=2,
    session_type=SessionType.A_CONTROLLED,
    punch_type=PunchType.JAB,
    intensity=Intensity.POWER,
    speed=Speed.FAST,
    num_punches=20,
    interval_range=(3.0, 6.0),
    duration_estimate_sec=120,
    instructions=[
        "Power jabs - maximum speed and power.",
        "",
        "ON EACH BEEP:",
        "  1. Throw ONE powerful jab - commit fully",
        "  2. Fast, explosive extension",
        "  3. Drive from the legs and rotate shoulder",
        "  4. Snap back to guard quickly",
        "",
        "Give each punch 100% effort.",
    ],
)

SESSION_06 = SessionConfig(
    session_id=6,
    name="jab_natural",
    day=2,
    session_type=SessionType.B_NATURAL,
    punch_type=PunchType.JAB,
    intensity=Intensity.MEDIUM,
    speed=Speed.MEDIUM,
    num_punches=25,
    interval_range=(4.0, 7.0),
    duration_estimate_sec=180,
    instructions=[
        "Jabs with natural movement - more realistic context.",
        "",
        "BETWEEN PUNCHES:",
        "  - Active footwork (bounce, shuffle)",
        "  - Head movement (slips, small rolls)",
        "  - Stay loose and mobile",
        "",
        "ON EACH BEEP:",
        "  1. Throw ONE medium jab from wherever you are",
        "  2. Continue moving after the punch",
        "",
        "Don't stop moving to throw - punch in motion.",
    ],
)

SESSION_07 = SessionConfig(
    session_id=7,
    name="jab_mixed",
    day=2,
    session_type=SessionType.B_NATURAL,
    punch_type=PunchType.JAB,
    intensity=Intensity.MIXED,
    speed=Speed.MIXED,
    num_punches=25,
    interval_range=(4.0, 7.0),
    duration_estimate_sec=180,
    instructions=[
        "Jabs with varied intensity - mix it up.",
        "",
        "Vary your punches naturally:",
        "  - Some light (setup jabs)",
        "  - Some medium (working jabs)",
        "  - Some power (hurt jabs)",
        "",
        "ON EACH BEEP:",
        "  1. Throw ONE jab - any intensity you choose",
        "  2. Mix up your power levels",
        "  3. Keep moving between punches",
    ],
)

SESSION_08 = SessionConfig(
    session_id=8,
    name="jab_power_natural",
    day=2,
    session_type=SessionType.B_NATURAL,
    punch_type=PunchType.JAB,
    intensity=Intensity.POWER,
    speed=Speed.FAST,
    num_punches=20,
    interval_range=(4.0, 7.0),
    duration_estimate_sec=180,
    instructions=[
        "Power jabs with natural movement.",
        "",
        "BETWEEN PUNCHES:",
        "  - Stay mobile, active footwork",
        "  - Move like you're in a fight",
        "",
        "ON EACH BEEP:",
        "  1. Throw ONE powerful jab",
        "  2. Full commitment - fast and hard",
        "  3. Continue moving after",
    ],
)

# ============================================================================
# DAY 3: HOOKS
# ============================================================================

SESSION_09 = SessionConfig(
    session_id=9,
    name="hook_light_slow",
    day=3,
    session_type=SessionType.A_CONTROLLED,
    punch_type=PunchType.HOOK,
    intensity=Intensity.LIGHT,
    speed=Speed.SLOW,
    num_punches=20,
    interval_range=(3.0, 6.0),
    duration_estimate_sec=120,
    instructions=[
        "Light, slow hooks - focus on the arc motion.",
        "",
        "ON EACH BEEP:",
        "  1. Throw ONE light LEFT hook (lead hook)",
        "  2. Slow, controlled arc motion",
        "  3. Elbow at 90 degrees, palm facing you",
        "  4. Rotate hips and shoulders",
        "  5. Return to guard",
        "",
        "Focus on the ROTATION - this is what distinguishes hooks.",
        "Remember: Sensor is on LEFT wrist, throw LEFT hooks.",
    ],
)

SESSION_10 = SessionConfig(
    session_id=10,
    name="hook_medium",
    day=3,
    session_type=SessionType.A_CONTROLLED,
    punch_type=PunchType.HOOK,
    intensity=Intensity.MEDIUM,
    speed=Speed.MEDIUM,
    num_punches=20,
    interval_range=(3.0, 6.0),
    duration_estimate_sec=120,
    instructions=[
        "Medium intensity hooks - normal sparring power.",
        "",
        "ON EACH BEEP:",
        "  1. Throw ONE medium-power hook",
        "  2. Good rotation through the hips",
        "  3. Snap the punch - arc out and back",
        "  4. Return to guard",
    ],
)

SESSION_11 = SessionConfig(
    session_id=11,
    name="hook_power_fast",
    day=3,
    session_type=SessionType.A_CONTROLLED,
    punch_type=PunchType.HOOK,
    intensity=Intensity.POWER,
    speed=Speed.FAST,
    num_punches=20,
    interval_range=(3.0, 6.0),
    duration_estimate_sec=120,
    instructions=[
        "Power hooks - maximum speed and rotation.",
        "",
        "ON EACH BEEP:",
        "  1. Throw ONE powerful hook",
        "  2. Explosive hip rotation",
        "  3. Drive through the target",
        "  4. Fast return to guard",
        "",
        "Commit fully - these should feel powerful.",
    ],
)

SESSION_12 = SessionConfig(
    session_id=12,
    name="hook_natural",
    day=3,
    session_type=SessionType.B_NATURAL,
    punch_type=PunchType.HOOK,
    intensity=Intensity.MEDIUM,
    speed=Speed.MEDIUM,
    num_punches=25,
    interval_range=(4.0, 7.0),
    duration_estimate_sec=180,
    instructions=[
        "Hooks with natural movement.",
        "",
        "BETWEEN PUNCHES:",
        "  - Active footwork",
        "  - Head movement",
        "  - Stay loose",
        "",
        "ON EACH BEEP:",
        "  1. Throw ONE medium hook from wherever you are",
        "  2. Continue moving after",
    ],
)

SESSION_13 = SessionConfig(
    session_id=13,
    name="hook_mixed",
    day=3,
    session_type=SessionType.B_NATURAL,
    punch_type=PunchType.HOOK,
    intensity=Intensity.MIXED,
    speed=Speed.MIXED,
    num_punches=25,
    interval_range=(4.0, 7.0),
    duration_estimate_sec=180,
    instructions=[
        "Hooks with varied intensity.",
        "",
        "Mix your hooks:",
        "  - Some light (checking hooks)",
        "  - Some medium (working hooks)",
        "  - Some power (knockout hooks)",
        "",
        "ON EACH BEEP:",
        "  1. Throw ONE hook - vary the intensity",
        "  2. Keep moving between punches",
    ],
)

SESSION_14 = SessionConfig(
    session_id=14,
    name="hook_power_natural",
    day=3,
    session_type=SessionType.B_NATURAL,
    punch_type=PunchType.HOOK,
    intensity=Intensity.POWER,
    speed=Speed.FAST,
    num_punches=20,
    interval_range=(4.0, 7.0),
    duration_estimate_sec=180,
    instructions=[
        "Power hooks with natural movement.",
        "",
        "ON EACH BEEP:",
        "  1. Throw ONE powerful hook",
        "  2. Full rotation, full commitment",
        "  3. Continue moving after",
    ],
)

# ============================================================================
# DAY 4: MIXED
# ============================================================================

SESSION_15 = SessionConfig(
    session_id=15,
    name="mixed",
    day=4,
    session_type=SessionType.C_MIXED,
    punch_type=PunchType.MIXED,
    intensity=Intensity.MEDIUM,
    speed=Speed.MEDIUM,
    num_punches=20,
    interval_range=(4.0, 6.0),
    duration_estimate_sec=180,
    instructions=[
        "Mixed punches - jabs and hooks randomized.",
        "",
        "Voice will announce JAB or HOOK before each beep.",
        "",
        "ON EACH BEEP:",
        "  1. Listen for announcement (JAB or HOOK)",
        "  2. Throw the announced punch",
        "  3. Medium power, good technique",
        "  4. Return to guard",
        "",
        "10 jabs and 10 hooks in random order.",
    ],
)

SESSION_16 = SessionConfig(
    session_id=16,
    name="mixed",
    day=4,
    session_type=SessionType.C_MIXED,
    punch_type=PunchType.MIXED,
    intensity=Intensity.MIXED,
    speed=Speed.MIXED,
    num_punches=20,
    interval_range=(4.0, 6.0),
    duration_estimate_sec=180,
    instructions=[
        "Mixed punches with varied intensity.",
        "",
        "Voice will announce JAB or HOOK before each beep.",
        "",
        "ON EACH BEEP:",
        "  1. Listen for announcement",
        "  2. Throw the announced punch",
        "  3. Vary your intensity naturally",
        "",
        "Move between punches - stay active.",
    ],
)

# ============================================================================
# DAY 5: VALIDATION (HOLD-OUT)
# ============================================================================

SESSION_17 = SessionConfig(
    session_id=17,
    name="holdout_jabs",
    day=5,
    session_type=SessionType.B_NATURAL,
    punch_type=PunchType.JAB,
    intensity=Intensity.MIXED,
    speed=Speed.MIXED,
    num_punches=50,
    interval_range=(3.0, 6.0),
    duration_estimate_sec=300,
    instructions=[
        "HOLD-OUT TEST SET - Jabs only.",
        "",
        "This data will NOT be used for training.",
        "It tests how well the model generalizes.",
        "",
        "ON EACH BEEP:",
        "  1. Throw ONE jab",
        "  2. Vary intensity naturally",
        "  3. Stay mobile between punches",
        "",
        "50 jabs total - pace yourself.",
    ],
    notes="HOLD-OUT SET - Do not use for training.",
)

SESSION_18 = SessionConfig(
    session_id=18,
    name="holdout_hooks",
    day=5,
    session_type=SessionType.B_NATURAL,
    punch_type=PunchType.HOOK,
    intensity=Intensity.MIXED,
    speed=Speed.MIXED,
    num_punches=50,
    interval_range=(3.0, 6.0),
    duration_estimate_sec=300,
    instructions=[
        "HOLD-OUT TEST SET - Hooks only.",
        "",
        "This data will NOT be used for training.",
        "",
        "ON EACH BEEP:",
        "  1. Throw ONE hook",
        "  2. Vary intensity naturally",
        "  3. Stay mobile between punches",
        "",
        "50 hooks total - pace yourself.",
    ],
    notes="HOLD-OUT SET - Do not use for training.",
)

SESSION_19 = SessionConfig(
    session_id=19,
    name="blind_mixed",
    day=5,
    session_type=SessionType.C_MIXED,
    punch_type=PunchType.MIXED,
    intensity=Intensity.MIXED,
    speed=Speed.MIXED,
    num_punches=20,
    interval_range=(4.0, 6.0),
    duration_estimate_sec=180,
    instructions=[
        "BLIND TEST - Labels hidden during evaluation.",
        "",
        "Voice will announce JAB or HOOK.",
        "",
        "ON EACH BEEP:",
        "  1. Throw the announced punch",
        "  2. Natural intensity and movement",
        "",
        "This tests the model's real-world performance.",
    ],
    notes="BLIND TEST - Labels recorded separately for evaluation.",
)

# ============================================================================
# DAY 6: STRESS TESTING
# ============================================================================

SESSION_20 = SessionConfig(
    session_id=20,
    name="fatigued",
    day=6,
    session_type=SessionType.B_NATURAL,
    punch_type=PunchType.MIXED,
    intensity=Intensity.MIXED,
    speed=Speed.MIXED,
    num_punches=30,
    interval_range=(3.0, 5.0),
    duration_estimate_sec=180,
    instructions=[
        "FATIGUED PUNCHES - After 5 minutes of exercise.",
        "",
        "BEFORE THIS SESSION:",
        "  - Do 5 minutes of intense shadowboxing",
        "  - Or 50 burpees / jumping jacks",
        "  - You should be breathing hard",
        "",
        "Voice will announce JAB or HOOK.",
        "",
        "ON EACH BEEP:",
        "  1. Throw the announced punch",
        "  2. Your form may degrade - that's expected",
        "  3. Keep going even if tired",
    ],
    pre_checklist=COMMON_CHECKLIST + ["Complete 5-minute intense warm-up first"],
    notes="Tests model robustness to fatigued/sloppy form.",
)

SESSION_21 = SessionConfig(
    session_id=21,
    name="hard_negatives",
    day=6,
    session_type=SessionType.D_HARD_NEGATIVE,
    punch_type=PunchType.NONE,
    intensity=Intensity.NA,
    speed=Speed.NA,
    num_punches=0,  # These are NOT punches
    interval_range=(3.0, 5.0),
    duration_estimate_sec=180,
    instructions=[
        "HARD NEGATIVES - Movements that are NOT punches.",
        "",
        "These movements could fool the model into false positives.",
        "",
        "Voice will announce what movement to do:",
        "  - REACH: Reach forward (like grabbing something)",
        "  - SWAT: Swatting motion to the side",
        "  - FACE: Touch your face / adjust headgear",
        "  - WIPE: Wipe sweat from forehead",
        "  - POINT: Point at something",
        "  - PARRY: Defensive parry motion",
        "  - DRINK: Reach for and drink water",
        "",
        "DO NOT throw actual punches in this session.",
    ],
    notes="Critical for reducing false positives.",
)

SESSION_22 = SessionConfig(
    session_id=22,
    name="edge_cases",
    day=6,
    session_type=SessionType.SPECIAL,
    punch_type=PunchType.MIXED,
    intensity=Intensity.MIXED,
    speed=Speed.MIXED,
    num_punches=30,
    interval_range=(2.0, 4.0),
    duration_estimate_sec=180,
    instructions=[
        "EDGE CASES - Combinations and feints.",
        "",
        "This session includes:",
        "  - Double jabs (jab-jab)",
        "  - 1-2 combinations (jab-cross, count as jab)",
        "  - Feints (fake punches - don't extend fully)",
        "  - Interrupted punches (start then stop)",
        "",
        "Voice will announce what to do.",
        "Listen carefully - some are combinations.",
    ],
    notes="Tests segmentation and edge case handling.",
)


# ============================================================================
# DAY 7: TEST
# ============================================================================

SESSION_23 = SessionConfig(
    session_id=23,
    name="test_mixed",
    day=7,
    session_type=SessionType.C_MIXED,
    punch_type=PunchType.MIXED,
    intensity=Intensity.MIXED,
    speed=Speed.MIXED,
    num_punches=20,
    interval_range=(3.0, 5.0),
    duration_estimate_sec=120,
    instructions=[
        "TEST SESSION - 10 jabs and 10 hooks randomized.",
        "",
        "Voice will announce JAB or HOOK before each beep.",
        "",
        "ON EACH BEEP:",
        "  1. Listen for announcement (JAB or HOOK)",
        "  2. Throw the announced punch",
        "  3. Natural intensity and movement",
        "  4. Return to guard",
        "",
        "This data will be used to test the trained model.",
    ],
    notes="Test session for model evaluation.",
)


# ============================================================================
# ALL SESSIONS REGISTRY
# ============================================================================

ALL_SESSIONS = [
    SESSION_00, SESSION_01, SESSION_02,  # Day 1
    SESSION_03, SESSION_04, SESSION_05, SESSION_06, SESSION_07, SESSION_08,  # Day 2
    SESSION_09, SESSION_10, SESSION_11, SESSION_12, SESSION_13, SESSION_14,  # Day 3
    SESSION_15, SESSION_16,  # Day 4
    SESSION_17, SESSION_18, SESSION_19,  # Day 5
    SESSION_20, SESSION_21, SESSION_22,  # Day 6
    SESSION_23,  # Day 7
]

SESSIONS_BY_DAY = {
    1: [SESSION_00, SESSION_01, SESSION_02],
    2: [SESSION_03, SESSION_04, SESSION_05, SESSION_06, SESSION_07, SESSION_08],
    3: [SESSION_09, SESSION_10, SESSION_11, SESSION_12, SESSION_13, SESSION_14],
    4: [SESSION_15, SESSION_16],
    5: [SESSION_17, SESSION_18, SESSION_19],
    6: [SESSION_20, SESSION_21, SESSION_22],
    7: [SESSION_23],
}

SESSIONS_BY_ID = {s.session_id: s for s in ALL_SESSIONS}

DAY_NAMES = {
    1: "Baseline & Calibration",
    2: "Jabs",
    3: "Hooks",
    4: "Mixed",
    5: "Validation (Hold-out)",
    6: "Stress Testing",
    7: "Test",
}

DAY_FOLDERS = {
    1: "day1_baseline",
    2: "day2_jabs",
    3: "day3_hooks",
    4: "day4_mixed",
    5: "day5_validation",
    6: "day6_stress",
    7: "day7_test",
}


def get_session(session_id: int) -> Optional[SessionConfig]:
    """Get session configuration by ID"""
    return SESSIONS_BY_ID.get(session_id)


def get_day_sessions(day: int) -> List[SessionConfig]:
    """Get all sessions for a specific day"""
    return SESSIONS_BY_DAY.get(day, [])


def print_session_summary():
    """Print summary of all sessions"""
    print("\n" + "="*70)
    print("  BOXING DATA COLLECTION - SESSION SUMMARY")
    print("="*70)

    for day in range(1, 8):
        sessions = get_day_sessions(day)
        print(f"\nDAY {day}: {DAY_NAMES[day]}")
        print("-" * 50)
        for s in sessions:
            punch_info = f"{s.punch_type.value}" if s.punch_type != PunchType.NONE else "none"
            print(f"  Session {s.session_id:2d}: {s.name:<20} | {s.session_type.value} | {punch_info:<6} | {s.num_punches:2d} punches")

    print("\n" + "="*70)
    total_punches = sum(s.num_punches for s in ALL_SESSIONS if s.punch_type != PunchType.NONE)
    print(f"  TOTAL PUNCHES: {total_punches}")
    print("="*70 + "\n")


if __name__ == "__main__":
    print_session_summary()
