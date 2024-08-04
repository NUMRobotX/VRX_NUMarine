from setuptools import find_packages, setup

package_name = 'gps_waypoint_logger'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/gps_waypoints.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nuc1',
    maintainer_email='c3339567@uon.edu.au',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gps_waypoint_logger = gps_waypoint_logger.gps_waypoint_logger:main',
            'logged_waypoint_follower = gps_waypoint_logger.logged_waypoint_follower:main',
            'gps_waypoint_publisher = gps_waypoint_logger.gps_waypoint_publisher:main'
        ],
    },
)
