.PHONY: test
test:
	python -m unittest test/test.py

documentation:
	cd doc && $(MAKE) html
