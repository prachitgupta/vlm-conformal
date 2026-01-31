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
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
import numpy as np


class MissionExecutor:
    """High-level mission executor for PX4 drones"""
    
    def __init__(self, connection_string="udp://:14540"):
        self.drone = System()
        self.connection_string = connection_string
        self.mission_state = "IDLE"
        self.current_position = np.zeros(3)
        self.goal_position = np.array([15.0, 10.0, 2.0])  # NED
        self.goal_reached = False
        
    async def connect(self):
        """Connect to the drone"""
        print(f"Connecting to drone at {self.connection_string}...")
        await self.drone.connect(system_address=self.connection_string)
        
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("✅ Connected to drone")
                break
        
        # Wait for position estimate
        print("Waiting for position estimate...")
        async for health in self.drone.telemetry.health():
            if health.is_local_position_ok:
                print("✅ Position estimate OK")
                break
    
    async def preflight_checks(self):
        """Perform preflight checks"""
        print("\n🔍 Preflight Checks:")
        
        async for health in self.drone.telemetry.health():
            print(f"  GPS: {'✅' if health.is_global_position_ok else '❌'}")
            print(f"  Gyro: {'✅' if health.is_gyrometer_calibration_ok else '❌'}")
            print(f"  Accel: {'✅' if health.is_accelerometer_calibration_ok else '❌'}")
            print(f"  Mag: {'✅' if health.is_magnetometer_calibration_ok else '❌'}")
            print(f"  Local Pos: {'✅' if health.is_local_position_ok else '❌'}")
            print(f"  Home Pos: {'✅' if health.is_home_position_ok else '❌'}")
            
            if health.is_local_position_ok and health.is_gyrometer_calibration_ok:
                return True
            break
        
        return False
    
    async def arm_and_takeoff(self, altitude=2.5):
        """Arm and takeoff"""
        print(f"\n🚁 Arming and taking off to {altitude}m...")
        
        # Set takeoff altitude
        await self.drone.action.set_takeoff_altitude(altitude)
        
        # Arm
        self.mission_state = "ARMING"
        await self.drone.action.arm()
        print("  Armed")
        
        # Takeoff
        self.mission_state = "TAKEOFF"
        await self.drone.action.takeoff()
        
        # Wait for altitude
        async for position in self.drone.telemetry.position():
            altitude_reached = position.relative_altitude_m > altitude * 0.95
            
            if altitude_reached:
                print(f"  ✅ Reached altitude: {position.relative_altitude_m:.2f}m")
                self.mission_state = "HOVERING"
                break
            
            await asyncio.sleep(0.1)
        
        # Extra stabilization time
        await asyncio.sleep(2)
    
    async def enable_offboard_mode(self):
        """Enable offboard control mode"""
        print("\n📡 Enabling offboard mode...")
        
        # Set initial setpoint
        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        )
        
        try:
            await self.drone.offboard.start()
            print("  ✅ Offboard mode enabled")
            self.mission_state = "OFFBOARD"
            return True
        except OffboardError as e:
            print(f"  ❌ Failed to enable offboard: {e}")
            return False
    
    async def monitor_mission(self):
        """Monitor mission progress"""
        print("\n📊 Monitoring mission...")
        
        async for position in self.drone.telemetry.position():
            # Update current position
            self.current_position = np.array([
                position.latitude_deg,  # For simulation this is local N
                position.longitude_deg, # For simulation this is local E
                -position.relative_altitude_m  # NED frame
            ])
            
            # Calculate distance to goal
            dist = np.linalg.norm(self.goal_position - self.current_position)
            
            # Check if goal reached
            if dist < 0.5 and not self.goal_reached:
                self.goal_reached = True
                print(f"\n🎯 Goal reached! Distance: {dist:.2f}m")
                self.mission_state = "GOAL_REACHED"
                break
            
            # Status update
            if not self.goal_reached:
                print(f"  Distance to goal: {dist:.2f}m | "
                      f"Alt: {position.relative_altitude_m:.2f}m",
                      end='\r')
            
            await asyncio.sleep(0.5)
    
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
    """Complete mission for simulation"""
    executor = MissionExecutor(connection_string="udp://:14540")
    
    try:
        # Connect
        await executor.connect()
        
        # Preflight
        if not await executor.preflight_checks():
            print("❌ Preflight checks failed")
            return
        
        # Arm and takeoff
        await executor.arm_and_takeoff(altitude=2.5)
        
        # Enable offboard
        if not await executor.enable_offboard_mode():
            print("❌ Failed to enable offboard mode")
            await executor.land_at_position()
            return
        
        print("\n🎮 MPC Controller should now be running in another terminal")
        print("   The drone will navigate to the goal autonomously")
        print("   Press Ctrl+C to abort mission\n")
        
        # Monitor mission
        await executor.monitor_mission()
        
        # Land at goal
        await executor.land_at_position()
        
        print("\n✅ Mission completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Mission aborted by user")
        await executor.land_at_position()
    except Exception as e:
        print(f"\n❌ Mission failed: {e}")
        await executor.emergency_stop()


async def run_hardware_mission():
    """Complete mission for real hardware (Starling 2)"""
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
        await executor.arm_and_takeoff(altitude=2.5)
        await executor.enable_offboard_mode()
        
        print("\n🎮 MPC Controller active - autonomous navigation")
        await executor.monitor_mission()
        
        # Land at goal or RTL
        response = input("\nLand here or Return to Launch? (land/rtl): ")
        if response.lower() == 'rtl':
            await executor.return_and_land()
        else:
            await executor.land_at_position()
        
        print("\n✅ Mission completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Emergency landing initiated")
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
    print("PX4 MISSION EXECUTOR")
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