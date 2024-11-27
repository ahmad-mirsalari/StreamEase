from setuptools import setup, find_packages

setup(
    name="streamease",          # Replace with your package name
    version="0.1.0",                   # Initial version
    packages=find_packages(),          # Automatically find packages in your directory
    install_requires=[                 # List your dependencies here
            "numpy",
            "onnx",
            "onnxoptimizer",
            # Add more dependencies as needed
    ],
    author="Seyed Ahmad Mirsalari",
    author_email="seyedahmad.mirsalar2@unibo.it",
    description="A brief description of your package",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/ahmad-mirsalari/TCN_Pre_release",  # Replace with your project's URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',            # Specify the Python versions supported
)
