from setuptools import setup, find_packages

setup(
    name="colorful_canvas",
    version="0.1.0",
    description="AI Art Studio for creating 3D visual illusions",
    author="Colorful Canvas Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "opencv-python>=4.8.0.76",
        "torch>=2.0.1",
        "transformers>=4.30.2",
        "pillow>=10.0.0",
        "matplotlib>=3.7.2",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 