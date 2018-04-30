LONG_DESCRIPTION = """
This package provides a Python tool to crawl html content, extract keywords against the corpus, and provide trending analysis.
It is developed solely for the use of Pitchbook Data, all rights reserved for the project's respective owners
"""

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


DESCRIPTION         = "trending_news_keyword: financial and tech trending and emerging keywords from news articles"
NAME                = "trending_news_keyword"
PACKAGES            = find_packages()
#PACKAGE_DATA        = {'movie_badger': ['examples/*.ipynb']}
AUTHOR              = "Anna Huang | Laurie Lai | Sheryl Wang | Su Wang"
AUTHOR_EMAIL        = "..../rhuang92@uw.edu/@uw.edu"
URL                 = 'https://github.com/andurilhuang/trending_news_keyword'
DOWNLOAD_URL        = 'https://github.com/andurilhuang/trending_news_keywords'
LICENSE             = 'MIT'
INSTALL_REQUIRES    = ['requests',
                       'pandas',
                       'numpy',
                       'json',
                       'datetime',
                       'newspaper',
                       'boto3',
                       'langdetect',
                       'nltk',
                       'threading',
                       'bs4',
                       'decimal',
                       'urllib',
                       're',
                       'forex_python',
                       'MySQLdb',
                       'itertools',
                       'string',
                       'collections',
                       'fuzzywuzzy',
                       'pycountry',
                       'calendar',
                       'math',
                       'csv']
VERSION             = '1.0.0'
KEYWORD             = 'movie revenue prediction'


setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=PACKAGES,
      #package_data=PACKAGE_DATA,
      install_requires=INSTALL_REQUIRES,
      keyworks=KEYWORD,
      classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'],
     )


