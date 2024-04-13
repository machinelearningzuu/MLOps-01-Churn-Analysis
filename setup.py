from typing import List
from setuptools import find_packages,setup

"""HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements"""

setup(
    name='TelcoChurn',
    version='0.0.1',
    author='machine learning zuu',
    author_email='isurualagiyawanna9717@gmail.com',
    install_requires=[
                    "scikit-learn",
                    "pandas",
                    "numpy"
                    ],
    packages=find_packages()
    )