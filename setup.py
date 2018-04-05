LONG_DESCRIPTION = """

This package provides a Python tool for building visualizations and predicting the revenue of
a movie based on the data related to from TMDB and OMDB database. 


"""

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


DESCRIPTION         = "movie_badger: Analysis and visualization of the correlation between movie infomation and movie revenue."
NAME                = "movie_badger"
PACKAGES            = find_packages()
#PACKAGE_DATA        = {'movie_badger': ['examples/*.ipynb']}
AUTHOR              = "Anna Huang | Chen Ren | Jingyun Chen | Junmeng Zhu | Weichen Xu"
AUTHOR_EMAIL        = "..../chenren7@uw.edu/@uw.edu"
URL                 = 'https://github.com/UWSEDS-aut17/uwseds-group-movie-badgers'
DOWNLOAD_URL        = 'https://github.com/UWSEDS-aut17/uwseds-group-movie-badgers'
LICENSE             = 'MIT'
INSTALL_REQUIRES    = ['requests','pandas','numpy','json','datetime','sklearn']
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


