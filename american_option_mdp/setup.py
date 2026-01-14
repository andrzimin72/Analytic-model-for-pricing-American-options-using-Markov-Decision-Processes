from setuptools import setup, find_packages

setup(
    name="american_option_mdp",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "mdptoolbox-python",
    ],
    python_requires=">=3.7",
)