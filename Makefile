.PHONY: test clean docs

# assume you have installed need packages
export SPHINX_MOCK_REQUIREMENTS=1

clean:
	# clean all temp runs
	rm -rf _ckpt_*
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf ./docs/build
	rm -rf ./docs/source/generated
	rm -rf ./docs/source/*/generated
	rm -rf ./docs/source/api

test: clean
	# Review the CONTRIBUTING docmentation for other ways to test.
	pip install -r requirements/devel.txt
	# install APEX, see https://github.com/NVIDIA/apex#linux

	# run tests with coverage
	python -m coverage run --source lightning_transformers -m pytest lightning_transformers tests -v
	python -m coverage report

docs: clean
	pip install --quiet -r requirements/docs.txt
	python -m sphinx -b html -W --keep-going docs/source docs/build
