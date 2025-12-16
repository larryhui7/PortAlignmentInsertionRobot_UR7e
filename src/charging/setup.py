from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'charging'

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
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ee106a-aic',
    maintainer_email='gabrielhan@berkekeley.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'move = charging.move:main',
            'cv = charging.cv:main',
            'toggle = charging.toggle_grip:main',
            'verify = charging.chessboardverify:main',
            'calibration = charging.calibration:main',
            'handeye = charging.handeyecalibration:main',
            'board = charging.boardbase:main',
            'aruco = charging.aruco:main',
            'spiral = charging.sprial:main',
        ],
    },
)
