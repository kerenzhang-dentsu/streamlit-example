from pathlib import Path

from setuptools import find_packages, setup

def read_requirements(path):
    return list(Path(path).read_text().splitlines())

base_reqs = read_requirements("/Users/kzhang10/Desktop/Github/streamlit-example/requirements.txt")

# orbit_reqs = read_requirements("requirements/orbit.txt")
# neuralprophet_reqs = read_requirements("requirements/neuralprophet.txt")

all_reqs = base_reqs

setup(
    name="test",
    version="1.0.1",
    description="An awesome adaptable framework to jumpstart your forecasting!",
    url="https://github.com/kerenzhang-dentsu/streamlit-example",
    maintainer="dentsu Media US Data Science Team",
    maintainer_email="",
    packages=find_packages(),
    install_requires=all_reqs,
    zip_safe=False
    )