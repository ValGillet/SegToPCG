from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Translate segmentation obtained from funkelab method to PyChunkedGraph graphene and protobuf format'
LONG_DESCRIPTION = ''

# Setting up
setup(
        name="segtopcg", 
        version=VERSION,
        author="Valentin Gillet",
        author_email="valentin.gillet@biol.lu.se",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['numpy', 'daisy'], 
        keywords=['python', 'segmentation']
    )