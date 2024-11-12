from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="predict-future-sales",
    version="1.1",
    author="valera2444",
    author_email="valerkaf2003.vf@gmail.com",
    description="Feature extraction, validation schema and hyperparameters tuning for future sales prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/valera2444/FutureSalesPrediction",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "optuna"
    ],
)