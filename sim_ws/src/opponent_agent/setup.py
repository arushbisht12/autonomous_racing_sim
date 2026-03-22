from setuptools import setup

package_name = 'opponent_agent'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Arush',
    maintainer_email='arush.bijay.bisht@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nice_opp = opponent_agent.nice_agent:main',
            'mean_opp = opponent_agent.mean_agent:main',
        ],
    },
)
