from setuptools import setup,find_packages


setup(
    name='diaghint',
    version='0.1.0',
    description='Experience-guided PostgreSQL query optimization',
    python_requires='>=3.11',
    package_dir={'':'src'},
    packages=find_packages('src'),
    install_requires=[
        'numpy>=1.24',
        'torch>=2.1',
        'scikit-learn>=1.3',
        'joblib>=1.3',
        'sqlglot>=25',
        'psycopg2-binary>=2.9',
        'openai>=1.40',
    ],
    entry_points={'console_scripts':['diaghint=diaghint.cli:main']},
)
