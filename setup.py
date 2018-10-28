import setuptools

with open("README.rst", "r") as f:
    long_description = f.read()

with open("openfermionblueqat/_version.py", "r") as f:
    exec(f.read())

setuptools.setup(
    name = "openfermionblueqat",
    version=__version__,
    author="MDR Inc.",
    author_email="kato@mdrft.com",
    description="Interface of OpenFermion with Blueqat",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/mdrft/OpenFermion-Blueqat",
    license="Apache 2",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 3 - Alpha",
    ]
)
