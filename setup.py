# setup.py is used to convert the ml project to single package. and to do pipipeline
# building an application 

from setuptools import find_packages,setup
from typing import List

# it(-e .) is in the end of the requirements file to automatic installations
HYPEN_E_DOT = '-e .'
def get_requirements(file_path: str)->List[str]:
    '''
    This function will return the list of requirements 
    '''
    # empty list to store the file library names
    requirements = []
    # to open the file
    with open(file_path) as file_obj:
        # read the lines and store in above list
        requirements = file_obj.readlines()
        # for the next line it includes \n -> we have delete it for proper installation
        requirements = [req.replace("\n","") for req in requirements]
        
        # condition to remove the (-e .) -> it is used for automatic call and install packages
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    # finnally return the list of python library names to the install_requires 
    return requirements        
        
# setup function: info about the project 
            
setup(
    name='mlproject',
    version='0.0.1',
    author='Jerome',
    author_email='jeromejerry333@gmail.com',
    packages=find_packages(),
   # install_requires=['pandas','numpy','seaborn','matplotlib'],
    install_requires=get_requirements('requirements.txt'),
)

