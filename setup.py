from setuptools import setup, find_packages


setup(
    name='ColaLib',
    version='0.1.0',
    author='Your Name',
    author_email='youremail@example.com',
    description='A short description of your project',
    long_description="Description",
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my_package',
    # 下面是几个案例
    packages=find_packages(include=["ColaPCRModules*", "ColaOpen3D*", "ColaDatasets*", "ColaUtils*"])  # 该项目下所有包
)  


"""
python cola_setup.py bdist_wheel
cd dist
pip install ..

"""