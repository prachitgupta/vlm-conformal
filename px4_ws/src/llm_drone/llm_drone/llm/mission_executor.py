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
from functools import partial
from mavsdk import System, telemetry
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

print = partial(builtins.print, flush=True)


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

    async def get_relative_altitude_m(self):
        """Return current relative altitude in meters."""
        async for position in self.drone.telemetry.position():
            return float(position.relative_altitude_m)
        return 0.0

    async def wait_until_armed(self, timeout_s=8.0):
        """Wait until the vehicle reports armed."""
        deadline = asyncio.get_running_loop().time() + float(timeout_s)
        while asyncio.get_running_loop().time() < deadline:
            if await self.get_armed():
                return True
            await asyncio.sleep(0.2)
        return await self.get_armed()

    async def wait_until_in_air(self, timeout_s=20.0):
        """Wait until the vehicle reports airborne."""
        deadline = asyncio.get_running_loop().time() + float(timeout_s)
        while asyncio.get_running_loop().time() < deadline:
            if await self.get_in_air():
                return True
            await asyncio.sleep(0.2)
        return await self.get_in_air()

    async def wait_until_altitude(self, target_altitude_m, timeout_s=25.0, altitude_tolerance_m=0.35):
        """Wait until the vehicle reaches the requested relative altitude.

        In SITL the reported relative altitude often settles slightly below the
        requested takeoff altitude, so we accept a small absolute shortfall
        instead of requiring a strict percentage of the target.
        """
        deadline = asyncio.get_running_loop().time() + float(timeout_s)
        last_altitude_m = await self.get_relative_altitude_m()
        required_altitude_m = max(0.0, float(target_altitude_m) - float(altitude_tolerance_m))
        while asyncio.get_running_loop().time() < deadline:
            last_altitude_m = await self.get_relative_altitude_m()
            if last_altitude_m >= required_altitude_m:
                return True, last_altitude_m
            await asyncio.sleep(0.2)
        return False, last_altitude_m

    async def ensure_hold_mode(self, timeout_s=8.0):
        """Best-effort: force HOLD before arming if the vehicle isn't already there."""
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

    async def arm_with_retry(self, max_attempts=3):
        """Arm with a HOLD-mode recovery path for transient command denials."""
        last_error = None
        for attempt in range(1, int(max_attempts) + 1):
            if attempt == 1:
                await self.ensure_hold_mode()
            try:
                await self.drone.action.arm()
                if await self.wait_until_armed(timeout_s=6.0):
                    return
                raise RuntimeError("Arm command returned, but vehicle never reported armed")
            except Exception as exc:
                last_error = exc
                print(f"  ⚠️ Arm attempt {attempt}/{max_attempts} failed: {exc}")
                if attempt >= int(max_attempts):
                    break
                await self.ensure_hold_mode()
                await asyncio.sleep(1.0)

        if last_error is not None:
            raise last_error
    
    async def arm_and_takeoff(self, altitude=1.5, takeoff_timeout_s=25.0, altitude_tolerance_m=0.35):
        """Arm and takeoff, failing fast if the climb never starts/finishes."""
        print(f"\n🚁 Arming and taking off to {altitude}m...")
        
        # Set takeoff altitude
        await self.drone.action.set_takeoff_altitude(altitude)
        
        # Arm
        self.mission_state = "ARMING"
        await self.arm_with_retry()
        print("  Armed")
        
        # Takeoff
        self.mission_state = "TAKEOFF"
        await self.drone.action.takeoff()

        if not await self.wait_until_in_air(timeout_s=min(10.0, float(takeoff_timeout_s))):
            raise RuntimeError("Takeoff command accepted, but vehicle never reported airborne")

        altitude_reached, latest_altitude_m = await self.wait_until_altitude(
            altitude,
            timeout_s=float(takeoff_timeout_s),
            altitude_tolerance_m=float(altitude_tolerance_m),
        )
        if not altitude_reached:
            required_altitude_m = max(0.0, float(altitude) - float(altitude_tolerance_m))
            raise RuntimeError(
                f"Takeoff timed out before reaching altitude {float(altitude):.2f}m "
                f"(required={required_altitude_m:.2f}m latest={latest_altitude_m:.2f}m)"
            )

        print(
            "  ✅ Reached takeoff altitude window: "
            f"{latest_altitude_m:.2f}m "
            f"(target={float(altitude):.2f}m tolerance={float(altitude_tolerance_m):.2f}m)"
        )
        self.mission_state = "HOVERING"
        
        # Extra stabilization time
        await asyncio.sleep(2)
    
    async def enable_offboard_mode(self, max_attempts=3):
        """Enable offboard control mode, retrying transient PX4 denials."""
        print("\n📡 Enabling offboard mode...")
        last_error = None
        for attempt in range(1, int(max_attempts) + 1):
            try:
                await self.drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                )
                await asyncio.sleep(0.1)
                print(f"  Sending OFFBOARD start request (attempt {attempt}/{max_attempts})")
                await self.drone.offboard.start()
                print("  ✅ Offboard mode enabled")
                self.mission_state = "OFFBOARD"
                return True
            except OffboardError as exc:
                last_error = exc
                print(f"  ⚠️ Offboard attempt {attempt}/{max_attempts} failed: {exc}")
                if attempt >= int(max_attempts):
                    break
                await asyncio.sleep(1.0)

        if last_error is not None:
            print(f"  ❌ Failed to enable offboard: {last_error}")
        return False

    async def recover_for_retry(self):
        """Try to reset PX4 into a clean state before the next setup attempt."""
        print("  ↺ Resetting vehicle state before retry...")
        try:
            await self.drone.offboard.stop()
        except Exception:
            pass

        if await self.get_in_air():
            print("  Landing after failed setup attempt...")
            try:
                await self.drone.action.land()
            except Exception as exc:
                print(f"  ⚠️ Land command failed during recovery: {exc}")
            deadline = asyncio.get_running_loop().time() + 30.0
            while asyncio.get_running_loop().time() < deadline:
                if not await self.get_in_air():
                    break
                await asyncio.sleep(0.5)

        if await self.get_armed():
            print("  Disarming after failed setup attempt...")
            try:
                await self.drone.action.disarm()
            except Exception as exc:
                print(f"  ⚠️ Disarm failed during recovery: {exc}")
            await asyncio.sleep(1.0)

        await self.ensure_hold_mode()
        await asyncio.sleep(1.0)

    async def prepare_for_external_offboard(
        self,
        *,
        altitude=2.5,
        setup_attempts=4,
        takeoff_timeout_s=25.0,
        altitude_tolerance_m=0.35,
    ):
        """Retry the full arm/takeoff/offboard sequence before giving up."""
        last_error = None
        for attempt in range(1, int(setup_attempts) + 1):
            print(f"\n🔁 Setup attempt {attempt}/{setup_attempts}")
            try:
                await self.arm_and_takeoff(
                    altitude=altitude,
                    takeoff_timeout_s=takeoff_timeout_s,
                    altitude_tolerance_m=altitude_tolerance_m,
                )
                if not await self.enable_offboard_mode():
                    raise RuntimeError("Offboard start failed after retries")
                return
            except Exception as exc:
                last_error = exc
                print(f"  ⚠️ Setup attempt {attempt}/{setup_attempts} failed: {exc}")
                if attempt >= int(setup_attempts):
                    break
                await self.recover_for_retry()

        if last_error is not None:
            raise last_error
        raise RuntimeError("Vehicle setup failed before OFFBOARD readiness")
    
    async def wait_ready_for_mpc(self):
        """Keep vehicle in offboard and wait for external MPC setpoints."""
        print("\n✅ Drone is airborne and in OFFBOARD mode.")
        print("   MPC can now command waypoints/setpoints.")
        print("   Press Ctrl+C to stop and land.\n")
        while True:
            await asyncio.sleep(1.0)
    
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
        if not await executor.preflight_checks():
            print("❌ Preflight checks failed")
            return
        
        await executor.prepare_for_external_offboard(altitude=2.5)
        
        await executor.wait_ready_for_mpc()
        
    except KeyboardInterrupt:
        print("\n⚠️  Stopping mission executor, landing...")
        await executor.land_at_position()
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
        if not await executor.preflight_checks():
            print("❌ Preflight checks failed - DO NOT FLY")
            return
        
        # Get telemetry
        await executor.get_telemetry()
        
        # Confirm takeoff
        response = input("\n⚠️  REAL HARDWARE MODE - Proceed with takeoff? (yes/no): ")
        if response.lower() != 'yes':
            print("Mission cancelled")
            return
        
        # Execute mission
        await executor.prepare_for_external_offboard(altitude=1.5)
        
        await executor.wait_ready_for_mpc()
        
    except KeyboardInterrupt:
        print("\n⚠️  Stopping mission executor, landing...")
        await executor.land_at_position()
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
