import os
from setuptools import find_packages, setup

package_name = 'llm_drone'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', [
            'config/llm_prompt.txt',
            'config/voxl_llm_prompt.txt',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='prachit',
    maintainer_email='prachitgupta100@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mpc = llm_drone.mpc_vision_controller:main',
            'mpc_local_planner = llm_drone.mpc_local_planner:main',
            'mpc_optimal_planner = llm_drone.mpc_optimal_planner:main',
            'mpc_sim = llm_drone.mpc_single_integrator_sim:main',
            'mission_executor = llm_drone.mission_executor:main',
            'llm = llm_drone.llm_planner:main',
            'mpc_voxl = llm_drone.voxl_mpc_controller:main',
            'llm_voxl = llm_drone.voxl_llm_planner:main',
            'performance_analyzer = llm_drone.performance_analyse:main',
            'dataset_generator = llm_drone.dataset_generator:main',
            'dataset_generator_sync = llm_drone.dataset_generator_sync:main',
            'debug_pointcloud_obstacles = llm_drone.debug_pointcloud_obstacles:main',
        ],
    },
)