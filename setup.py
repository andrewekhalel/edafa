from setuptools import setup


def readme():
	with open('README.md') as f:
		return f.read()


setup(name='edafa',
	version='0.1.1',
	description='Test Time Augmentation (TTA) wrapper for computer vision tasks: segmentation,classification, super-resolution, ... etc.',
	long_description=readme(),
	long_description_content_type="text/markdown",
	classifiers=[
	'Development Status :: 2 - Pre-Alpha',
	'License :: OSI Approved :: MIT License',
	'Operating System :: OS Independent',
	'Programming Language :: Python',
	'Programming Language :: Python :: 2',
	'Programming Language :: Python :: 2.6',
	'Programming Language :: Python :: 2.7',
	'Programming Language :: Python :: 3',
	'Programming Language :: Python :: 3.1',
	'Programming Language :: Python :: 3.2',
	'Programming Language :: Python :: 3.3',
	'Programming Language :: Python :: 3.4',
	'Programming Language :: Python :: 3.5',
	'Programming Language :: Python :: 3.6',
	],
	keywords='augmentation classification segmentation super-resolution pansharpening keras tensorflow pytorch',
	url='https://github.com/andrewekhalel/edafa',
	author='Andrew Khalel',
	author_email='andrewekhalel@gmail.com',
	license='MIT',
	packages=['edafa'],
	test_suite='nose.collector',
	tests_require=['nose'],
	install_requires=[
	'numpy', 'tifffile', 'opencv-python' 
	],
        zip_safe=False)
