from setuptools import setup, find_packages
from typing import List
import os


def get_requirements(file_path: str) -> List[str]:
    """
    Reads a requirements.txt file and returns a list of packages.
    Removes '-e .' if present.
    """
    requirements = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Requirements file not found at {file_path}")

    with open(file_path, "r", encoding="utf-8") as file_obj:
        requirements = [
            req.strip() for req in file_obj.readlines()
            if req.strip() and not req.startswith("#")  # ignore blank lines and comments
        ]

    # Remove editable install flag if present
    if "-e ." in requirements:
        requirements.remove("-e .")

    return requirements


setup(
    name="azure-project",
    version="1.0",
    author="Md Abdul Bari",
    author_email="abdulbari80@gmail.com",
    description="End-to-end predictive ML project with Flask web app and CI/CD deployment on Azure.",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.12",
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "Framework :: Flask",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)  
