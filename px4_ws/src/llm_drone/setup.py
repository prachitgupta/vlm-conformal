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
        'mission_executor = llm_drone.mission_executor:main',
        'llm = llm_drone.llm_planner:main',
        'mpc_voxl = llm_drone.voxl_mpc_controller:main',
        'llm_voxl = llm_drone.voxl_llm_planner:main',
        ],
    },
)
