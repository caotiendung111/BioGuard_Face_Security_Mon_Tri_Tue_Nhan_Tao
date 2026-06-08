# ============================================================
# DeepLock Pro (bioguard-face-security) — Makefile
# Common management and reproducibility shortcuts
# ============================================================

.PHONY: install run main test clean

# --- Setup ---
install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	@if not exist .env (copy .env.example .env)

# --- Execution ---
run:
	streamlit run app.py --server.port 8501

main:
	python main.py --camera 0

# --- Testing ---
test:
	python -m pytest tests/ -v

# --- Cleanup ---
clean:
	@if exist __pycache__ (rmdir /s /q __pycache__)
	@if exist tests\__pycache__ (rmdir /s /q tests\__pycache__)
	@if exist .pytest_cache (rmdir /s /q .pytest_cache)
	@if exist test_faces_temp (rmdir /s /q test_faces_temp)
	@echo Cleanup complete.
