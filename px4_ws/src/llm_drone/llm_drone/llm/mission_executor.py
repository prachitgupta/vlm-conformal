#!/usr/bin/env python3
"""
Mission Executor - High-level mission control
Handles takeoff, landing, and mission state machine
Works with both simulation and real hardware

Usage:
    python mission_executor.py --sim  # For simulation
    python mission_executor.py        # For real hardware
"""

import asyncio
import argparse
import builtins
import signal
from functools import partial
from mavsdk import System, telemetry
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

print = partial(builtins.print, flush=True)
TAKEOFF_COMPLETE_MARKER = "Takeoff completion: drone is airborne."
MISSION_READY_MARKER = "Drone is airborne and in OFFBOARD mode."


class MissionExecutor:
    """High-level mission executor for PX4 drones"""
    
    def __init__(self, connection_string="udp://:14540"):
        self.drone = System()
        self.connection_string = connection_string
        self.mission_state = "IDLE"

    async def _wait_for_connection_state(self, timeout_s=20.0):
        """Wait until MAVSDK reports a connected vehicle, with a timeout."""
        deadline = asyncio.get_running_loop().time() + float(timeout_s)
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                return state
            if asyncio.get_running_loop().time() >= deadline:
                break
        raise TimeoutError(f"Timed out waiting for MAVSDK connection on {self.connection_string}")

    async def _wait_for_local_position_estimate(self, timeout_s=30.0):
        """Wait until PX4 publishes a usable local position estimate."""
        deadline = asyncio.get_running_loop().time() + float(timeout_s)
        async for health in self.drone.telemetry.health():
            if health.is_local_position_ok:
                return health
            if asyncio.get_running_loop().time() >= deadline:
                break
        raise TimeoutError("Timed out waiting for local position estimate from PX4")
        
    async def connect(self):
        """Connect to the drone"""
        print(f"Connecting to drone at {self.connection_string}...")
        await self.drone.connect(system_address=self.connection_string)

        await self._wait_for_connection_state(timeout_s=20.0)
        print("✅ Connected to drone")

        # Wait for position estimate
        print("Waiting for position estimate...")
        await self._wait_for_local_position_estimate(timeout_s=30.0)
        print("✅ Position estimate OK")
    
    async def preflight_checks(self):
        """Perform preflight checks"""
        print("\n🔍 Preflight Checks:")

        try:
            health = await asyncio.wait_for(anext(self.drone.telemetry.health()), timeout=10.0)
        except TimeoutError:
            print("  ⚠️ Timed out waiting for telemetry.health() during preflight")
            return False

        print(f"  GPS: {'✅' if health.is_global_position_ok else '❌'}")
        print(f"  Gyro: {'✅' if health.is_gyrometer_calibration_ok else '❌'}")
        print(f"  Accel: {'✅' if health.is_accelerometer_calibration_ok else '❌'}")
        print(f"  Mag: {'✅' if health.is_magnetometer_calibration_ok else '❌'}")
        print(f"  Local Pos: {'✅' if health.is_local_position_ok else '❌'}")
        print(f"  Home Pos: {'✅' if health.is_home_position_ok else '❌'}")

        return health.is_local_position_ok and health.is_gyrometer_calibration_ok

    async def get_flight_mode(self):
        """Return the current PX4 flight mode."""
        async for mode in self.drone.telemetry.flight_mode():
            return mode
        return telemetry.FlightMode.UNKNOWN

    async def get_armed(self):
        """Return whether PX4 currently reports the vehicle as armed."""
        async for armed in self.drone.telemetry.armed():
            return bool(armed)
        return False

    async def get_in_air(self):
        """Return whether PX4 currently reports the vehicle as airborne."""
        async for in_air in self.drone.telemetry.in_air():
            return bool(in_air)
        return False

    async def wait_until_armed(self, timeout_s=8.0):
        """Wait until the vehicle reports armed."""
        deadline = asyncio.get_running_loop().time() + float(timeout_s)
        while asyncio.get_running_loop().time() < deadline:
            if await self.get_armed():
                return True
            await asyncio.sleep(0.2)
        return await self.get_armed()

    async def wait_until_disarmed(self, timeout_s=8.0):
        """Wait until the vehicle reports disarmed."""
        deadline = asyncio.get_running_loop().time() + float(timeout_s)
        while asyncio.get_running_loop().time() < deadline:
            if not await self.get_armed():
                return True
            await asyncio.sleep(0.2)
        return not await self.get_armed()

    async def wait_until_in_air(self, timeout_s=20.0):
        """Wait until the vehicle reports airborne."""
        deadline = asyncio.get_running_loop().time() + float(timeout_s)
        while asyncio.get_running_loop().time() < deadline:
            if await self.get_in_air():
                return True
            await asyncio.sleep(0.2)
        return await self.get_in_air()

    async def ensure_hold_mode(self, timeout_s=8.0):
        """Best-effort: switch to HOLD before arming if needed."""
        mode = await self.get_flight_mode()
        print(f"  Current flight mode before arming: {mode.name}")
        if mode == telemetry.FlightMode.HOLD:
            return True

        print("  Switching vehicle to HOLD before arming...")
        try:
            await self.drone.action.hold()
        except Exception as exc:
            print(f"  ⚠️ HOLD command failed: {exc}")
            return False

        deadline = asyncio.get_running_loop().time() + float(timeout_s)
        async for observed_mode in self.drone.telemetry.flight_mode():
            if observed_mode == telemetry.FlightMode.HOLD:
                print("  ✅ Vehicle is now in HOLD")
                return True
            if asyncio.get_running_loop().time() >= deadline:
                break

        latest_mode = await self.get_flight_mode()
        print(f"  ⚠️ Timed out waiting for HOLD; latest mode is {latest_mode.name}")
        return latest_mode == telemetry.FlightMode.HOLD

    async def arm(self):
        """Arm the vehicle."""
        print("\n🚁 Arming...")
        self.mission_state = "ARMING"
        if await self.get_armed():
            print("  ✅ Already armed")
            return
        await self.drone.action.arm()
        if not await self.wait_until_armed(timeout_s=6.0):
            raise RuntimeError("Arm command returned, but vehicle never reported armed")
        print("  ✅ Armed")

    async def disarm(self):
        """Disarm the vehicle if it is armed."""
        print("\n🔻 Disarming...")
        if not await self.get_armed():
            print("  ✅ Already disarmed")
            return
        await self.drone.action.disarm()
        if not await self.wait_until_disarmed(timeout_s=6.0):
            raise RuntimeError("Disarm command returned, but vehicle never reported disarmed")
        print("  ✅ Disarmed")

    async def takeoff(self, takeoff_timeout_s=20.0, stabilize_time_s=3.0):
        """Take off and wait until the drone is airborne and settled."""
        print("\n🛫 Taking off...")
        self.mission_state = "TAKEOFF"
        await self.drone.action.takeoff()
        if not await self.wait_until_in_air(timeout_s=float(takeoff_timeout_s)):
            raise RuntimeError("Takeoff command accepted, but vehicle never reported airborne")
        print("  ✅ Drone is airborne")
        print(f"[startup] {TAKEOFF_COMPLETE_MARKER}")
        self.mission_state = "HOVERING"
        if float(stabilize_time_s) > 0.0:
            print(f"  Stabilizing for {float(stabilize_time_s):.1f}s...")
            await asyncio.sleep(float(stabilize_time_s))

    async def enable_offboard_mode(self):
        """Enable offboard control mode."""
        print("\n📡 Enabling offboard mode...")
        try:
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
            )
            await asyncio.sleep(0.1)
            await self.drone.offboard.start()
        except OffboardError as exc:
            raise RuntimeError(f"Failed to enable offboard mode: {exc}") from exc
        print("  ✅ Offboard mode enabled")
        self.mission_state = "OFFBOARD"

    async def prepare_for_external_offboard(
        self,
        *,
        takeoff_timeout_s=20.0,
        stabilize_time_s=3.0,
    ):
        """Simple setup flow: HOLD, arm, take off, settle, then switch to OFFBOARD."""
        print("\n[startup] preparing vehicle for external offboard control")
        if not await self.ensure_hold_mode():
            raise RuntimeError("Failed to switch vehicle to HOLD before arming")
        await self.arm()
        await self.takeoff(
            takeoff_timeout_s=takeoff_timeout_s,
            stabilize_time_s=stabilize_time_s,
        )
        await self.enable_offboard_mode()

    async def preflight_checks_with_arm_retry(self):
        """Retry preflight once by disarming and re-arming if needed."""
        if await self.preflight_checks():
            return True

        print("\n⚠️  Preflight checks failed; disarming and retrying arm...")
        await self.disarm()
        if not await self.ensure_hold_mode():
            print("❌ Failed to switch to HOLD for preflight retry")
            return False
        await self.arm()
        return await self.preflight_checks()

    async def wait_for_interrupt_and_land(self):
        """Keep the process alive until interrupted, then land."""
        print(f"\n✅ {MISSION_READY_MARKER}")
        print("   MPC can now command waypoints/setpoints.")
        print("   Press Ctrl+C to land.\n")

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        installed_handlers = []

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
                installed_handlers.append(sig)
            except (NotImplementedError, RuntimeError):
                continue

        try:
            if installed_handlers:
                await stop_event.wait()
            else:
                while True:
                    await asyncio.sleep(1.0)
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            for sig in installed_handlers:
                loop.remove_signal_handler(sig)

        print("\n⚠️  Stopping mission executor, landing...")
        await self.land_at_position()
    
    async def return_and_land(self):
        """Return to launch and land"""
        print("\n🏠 Returning to launch position...")
        
        # Stop offboard mode
        try:
            await self.drone.offboard.stop()
        except:
            pass
        
        # Return to launch
        self.mission_state = "RTL"
        await self.drone.action.return_to_launch()
        
        # Wait for landing
        print("  Landing...")
        async for in_air in self.drone.telemetry.in_air():
            if not in_air:
                print("  ✅ Landed")
                self.mission_state = "LANDED"
                break
    
    async def land_at_position(self):
        """Land at current position"""
        print("\n🛬 Landing at current position...")
        
        # Stop offboard mode
        try:
            await self.drone.offboard.stop()
        except:
            pass
        
        self.mission_state = "LANDING"
        await self.drone.action.land()
        
        # Wait for landing
        async for in_air in self.drone.telemetry.in_air():
            if not in_air:
                print("  ✅ Landed")
                self.mission_state = "LANDED"
                break
    
    async def emergency_stop(self):
        """Emergency stop"""
        print("\n🚨 EMERGENCY STOP!")
        try:
            await self.drone.offboard.stop()
        except:
            pass
        await self.drone.action.kill()
        self.mission_state = "EMERGENCY"
    
    async def get_telemetry(self):
        """Get and print telemetry"""
        async for battery in self.drone.telemetry.battery():
            async for gps in self.drone.telemetry.gps_info():
                print(f"\n📡 Telemetry:")
                print(f"  Battery: {battery.remaining_percent*100:.0f}%")
                print(f"  GPS Satellites: {gps.num_satellites}")
                return


async def run_simulation_mission():
    """Prepare simulation drone for external MPC control."""
    executor = MissionExecutor(connection_string="udp://:14540")
    
    try:
        # Connect
        await executor.connect()
        
        # Preflight
        if not await executor.preflight_checks_with_arm_retry():
            print("❌ Preflight checks failed")
            return
        
        await executor.prepare_for_external_offboard()
        await executor.wait_for_interrupt_and_land()
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        await executor.emergency_stop()


async def run_hardware_mission():
    """Prepare real hardware drone for external MPC control."""
    # For real hardware, connection might be serial
    executor = MissionExecutor(connection_string="serial:///dev/ttyUSB0:921600")
    
    try:
        await executor.connect()
        
        # More thorough preflight for real hardware
        if not await executor.preflight_checks_with_arm_retry():
            print("❌ Preflight checks failed - DO NOT FLY")
            return
        
        # Get telemetry
        await executor.get_telemetry()
        
        # Confirm takeoff
        response = input("\n⚠️  REAL HARDWARE MODE - Proceed with takeoff? (yes/no): ")
        if response.lower() != 'yes':
            print("Mission cancelled")
            return
        
        await executor.prepare_for_external_offboard()
        await executor.wait_for_interrupt_and_land()
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        await executor.emergency_stop()


def main():
    parser = argparse.ArgumentParser(description='PX4 Mission Executor')
    parser.add_argument('--sim', action='store_true', 
                       help='Run in simulation mode')
    args = parser.parse_args()
    
    print("=" * 70)
    print("PX4 TAKEOFF + OFFBOARD PREP")
    print("=" * 70)
    
    if args.sim:
        print("Mode: SIMULATION")
        asyncio.run(run_simulation_mission())
    else:
        print("Mode: REAL HARDWARE (Starling 2)")
        print("⚠️  WARNING: This will control real hardware!")
        asyncio.run(run_hardware_mission())


if __name__ == '__main__':
    main()
