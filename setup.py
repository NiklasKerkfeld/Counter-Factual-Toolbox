from setuptools import setup, find_packages

setup(
    name="CounterFactualToolbox",
    version="1.0.0",
    packages=find_packages(include=["CounterFactualToolbox", "CounterFactualToolbox.*"]),
    install_requires=[
        "grad-cam >= 1.5.5",
        "matplotlib >= 3.10.1",
        "monai == 1.4.0",
        "numpy >= 1.26.4",
        "Pillow == 11.2.1",
        "torch == 2.6.0",
        "tqdm == 4.67.1"
    ],
    author="Niklas Kerkfeld",
    author_email="s6nikerk@uni-bonn.de",
    description="The CounterFactualToolbox is a toolbox to generate counter factual image.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NiklasKerkfeld/SegmentationBacktrackingSandbox",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.11",
)