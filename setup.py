import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

reqs = []
setuptools.setup(
    name='dp4imaging',
    version='0.1',
    author='Ali Siahkoohi',
    author_email='alisk@gatech.edu',
    description='Deep Bayesian inference for seismic imaging with tasks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/slimgroup/dp4imaging',
    license='MIT',
    install_requires=reqs,
    packages=setuptools.find_packages()
)
