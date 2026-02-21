from setuptools import setup, find_packages

setup(
    name="ragevision",
    version="1.0.0",
    description="CNN-driven emotion detection in streaming content",
    author="Mehrdad Momeni zadeh",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
        "opencv-python>=4.7.0",
    ],
)
