from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

## edit below variables as per your requirements -
REPO_NAME = "BookRecommendationProject"
AUTHOR_USER_NAME = "Akshita Tomar"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = ['streamlit','numpy']


setup(
    name=SRC_REPO,
    version="0.0.1",
    author="AKSHITA TOMAR",
    description="A small local packages for ML based books recommendations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AkshitaTomar1/BookRecommendationProject",
    author_email="axtomar7@gmail.com",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.7",
    install_requires=LIST_OF_REQUIREMENTS
)