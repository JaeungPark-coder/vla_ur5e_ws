import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'vla_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wodndqke',
    maintainer_email='wodndqke@gmail.com',
    description='ROS2 bridge running a fine-tuned openpi (pi0) VLA policy against a UR5e (real or Isaac Sim)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vla_policy_client = vla_bridge.vla_policy_client:main',
        ],
    },
)
