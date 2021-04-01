# This is a pretty crude example of what the setup.py may end up looking like
from setuptools import setup, find_packages


# Read License for display during setup, I believe we can exclude this as there is no license.
#with open("LICENSE FILE", "r") as fh:
#    passivlicense = fh.read()

# Read README for pypi long description
with open("README.md", "r") as fh:
    long_description = fh.read()

# Read requirements for requirements from file...
# There are different ways of specifying requirements for distribution vs. development.
# This assumes that the development and usage requirements are the same.
with open("requirements.txt", "r") as fh:
    requirements = fh.read()

# Begin Setup
setup(

    # Set the Author as @Enarjord
    author='@Enarjord',

    # Set the library name for 'pip install <name>'
    name='passivbot',

    # Set the version of this library
    version='2.1.0',

    # Set the website for the project
    url='https://github.com/enarjord/passivbot_futures/',

    # Set the License to License.md
    license="none",

    # Retrieve requirements from requirements.txt
    install_requires=[requirements],

    # Define the packages required for setup
    packages=find_packages(where=['passivbot', 'passivbot.*']),  # These are the container names of the package

    # Allow additional files to be packaged with the bot
    include_package_data=True,

    # Set the description
    description='Preferential grid trading framework for Binance and ByBit Perpetual Futures.',

    # Set the long description for pypi
    long_description=long_description,
    long_description_content_type="text/markdown",

    # Add additional information for SEO
    classifiers=[

        # Set intended audience
        'Intended Audience :: Financial and Cryptocurrency',

        # Set Custom License.
        'License :: Custom / None',

        # Set language
        'Natural Language :: English',

        # Set compatibility
        'Operating System :: OS Independent',

        # The client is intended for PYTHON 3.8
        'Programming Language :: Python :: 3.8 (Preferred) or above'
    ],

    # Set minimum python version
    python_requires='>=3.8',
)