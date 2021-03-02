import setuptools

info = {
    'packages': ['timple'],
    'install_requires': [
        'numpy', 'matplotlib',
    ],
    'license': 'MIT',
    'url': 'https://github.com/theOehrly/Timple',
    'description': 'A package that provides extended functionality for '
                   'plotting timedelta-like values with Matplotlib.',
    'zip_safe': False
}

setuptools.setup(**info)
