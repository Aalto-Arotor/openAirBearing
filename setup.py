from setuptools import setup, find_packages

setup(
    name="OpenAirBearing",
    version="0.1.0",
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        "dash",
        "numpy",
        "plotly",
        "scipy",
        "callbacks"
    ],
)