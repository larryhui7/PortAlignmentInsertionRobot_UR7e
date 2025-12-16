from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'loadcell'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py'))
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='larryhui',
    maintainer_email='larryhui@example.com',
    description='ROS2 package for interfacing with ESP32 loadcell via serial port',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'loadcell_node = loadcell.loadcell_node:main',
        ],
    },
)

