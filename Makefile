# Variables
VENV_NAME = venv
PYTHON = python3
PIP = $(VENV_NAME)/bin/pip
PYTHON_VENV = $(VENV_NAME)/bin/python

.PHONY: all install clean fclean re run predict run-args help-program help

# Default target
all: install

# Create virtual environment and install dependencies
install: $(VENV_NAME)/bin/activate

# Create virtual environment
$(VENV_NAME)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Run the main program with default arguments
run: install
	$(PYTHON_VENV) main.py

# Predict price for a given km value (usage: make predict KM=100000)
predict: install
	@if [ -z "$(KM)" ]; then \
		echo "Error: KM parameter is required. Usage: make predict KM=100000"; \
		exit 1; \
	fi
	$(PYTHON_VENV) main.py --prediction $(KM)

# Clean compiled Python files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Remove virtual environment and compiled files
fclean: clean
	rm -rf $(VENV_NAME)

# Rebuild everything
re: fclean all

# Show help for Makefile targets
help:
	@echo "Available targets:"
	@echo "  install      - Create virtual environment and install dependencies"
	@echo "  run          - Run with default arguments (train model and show plots)"
	@echo "  predict      - Predict price for given km (use KM=value)"
	@echo "  run-args     - Run with custom arguments (use ARGS=\"--arg1 val1 --arg2 val2\")"
	@echo "  help-program - Show program's command line help"
	@echo "  clean        - Remove compiled Python files"
	@echo "  fclean       - Remove virtual environment and compiled files"
	@echo "  re           - Rebuild everything (fclean + install)"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make predict KM=100000"
	@echo "  make predict KM=50000"
	@echo "  make run-args ARGS=\"--learning-rate 0.05 --prediction 75000 --no-plots\""
