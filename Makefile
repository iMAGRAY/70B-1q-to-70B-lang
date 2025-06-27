install:
	pip install -e .


test: install
	pytest -q


demo:
	python -m sigla demo


build-demo:
	cd cpp_glass_demo && ./build.sh
