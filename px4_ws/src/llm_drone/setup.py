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
            'config/llm_planner_prompt.txt',
            'config/variant_X.txt',
            'config/voxl_llm_prompt.txt',
        ]),
    ],
    install_requires=[
        'setuptools',
        'openai',
        'instructor',
        'pydantic',
    ],
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
            'mpc = llm_drone.mpc.mpc_vision_controller:main',
            'mpc_local_planner = llm_drone.mpc.mpc_local_planner:main',
            'mpc_optimal_planner = llm_drone.mpc.mpc_optimal_planner:main',
            'mpc_sim = llm_drone.mpc.mpc_single_integrator_sim:main',
            'mission_executor = llm_drone.llm.mission_executor:main',
            'llm = llm_drone.llm.llm_planner:main',
            'mpc_voxl = llm_drone.mpc.voxl_mpc_controller:main',
            'llm_voxl = llm_drone.llm.voxl_llm_planner:main',
            'performance_analyzer = llm_drone.verifier.performance_analyse:main',
            'dataset_generator = llm_drone.mpc.dataset_generator:main',
            'dataset_generator_sync = llm_drone.llm.dataset_generator_sync:main',
            'dataset_generator_executor = llm_drone.llm.dataset_generator_executor:main',
            'debug_pointcloud_obstacles = llm_drone.verifier.debug_pointcloud_obstacles:main',
            'eval_llm = llm_drone.verifier.eval:main',
            'trajectory_goto_debug = llm_drone.verifier.trajectory_goto_debug:main',
            'trajectory_goto_mode_debug = llm_drone.verifier.trajectory_goto_mode_debug:main',
            'generate_synthetic_prompts = llm_drone.llm.generate_synthetic_prompts:main',
            'ground_truth_generator = llm_drone.llm.ground_truth_generator:main',
            'offline_dataset_batch_runner = llm_drone.llm.offline_dataset_batch_runner:main',
            'live_env_prompt_monitor = llm_drone.llm.live_env_prompt_monitor:main',
        ],
    },
)
