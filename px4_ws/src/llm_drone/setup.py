import sys

from setuptools import find_packages, setup
from setuptools.command.develop import develop as _develop


def _strip_unsupported_develop_flags(argv):
    """Remove colcon-only develop flags unsupported by newer setuptools."""
    cleaned = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in ("--uninstall", "--editable"):
            continue
        if arg == "--build-directory":
            skip_next = True
            continue
        if arg.startswith("--build-directory="):
            continue
        cleaned.append(arg)
    return cleaned


if "develop" in sys.argv:
    # colcon invokes `setup.py develop --uninstall --editable --build-directory ...`
    # Newer setuptools no longer accepts these flags, so strip them.
    sys.argv = _strip_unsupported_develop_flags(sys.argv)


class develop(_develop):
    user_options = _develop.user_options + [
        ("script-dir=", None, "directory for scripts (compat with colcon)"),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.script_dir = None

    def finalize_options(self):
        super().finalize_options()
        if self.script_dir and not getattr(self, "install_dir", None):
            self.install_dir = self.script_dir

    def run(self):
        # Avoid pip build isolation (no network in CI); use current env deps.
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            ".",
            "--use-pep517",
            "--no-build-isolation",
            "--no-deps",
        ]
        if self.install_dir:
            cmd += ["--target", self.install_dir]
        if self.user:
            cmd.append("--user")
        if self.prefix:
            cmd += ["--prefix", self.prefix]
        if self.index_url:
            cmd += ["--index-url", self.index_url]
        import subprocess

        subprocess.check_call(cmd)

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
        'mission_executor = llm_drone.mission_executor:main',
        'llm = llm_drone.llm_planner:main',
        'mpc_voxl = llm_drone.voxl_mpc_controller:main',
        'llm_voxl = llm_drone.voxl_llm_planner:main',
        'performance_analyzer = llm_drone.performance_analyse:main',
        ],
    },
    cmdclass={
        'develop': develop,
    },
)
