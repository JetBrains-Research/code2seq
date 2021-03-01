from setuptools import setup, find_packages

VERSION = "0.0.2"

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as requirements_file:
    install_requires = [line.strip() for line in requirements_file]

setup_args = dict(
    name="code2seq",
    version=VERSION,
    description="Set of pytorch modules and utils to train code2seq model",
    long_description_content_type="text/markdown",
    long_description=readme,
    install_requires=install_requires,
    license="MIT",
    packages=find_packages(),
    author="Egor Spirin",
    author_email="spirin.egor@gmail.com",
    keywords=["code2seq", "pytorch", "pytorch-lightning", "ml4code", "ml4se"],
    url="https://github.com/JetBrains-Research/code2seq",
    download_url="https://pypi.org/project/code2seq/",
)

if __name__ == "__main__":
    setup(**setup_args)
