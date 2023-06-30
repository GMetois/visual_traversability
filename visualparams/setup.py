from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Parameters for the visual_traversability ROS Node'

# Setting up
setup(
       # The name must match the folder name 'params'
        name="visualparams", 
        version=VERSION,
        author="Gabriel Métois",
        author_email="<gabriel.metois@ensta-paris.fr>",
        description=DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # Add any additional packages that 
        # needs to be installed along with your package.
        
        keywords=['ROS', 'parameters', 'pytorch'],
        classifiers= [
            "Development Status :: 1 - Planning",
            "License :: OSI Approved :: MIT License",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            'Operating System :: POSIX :: Linux',
        ]
)

# To install this package, run "pip install ." in the terminal