:: Run all static analysis checks.
@echo off
cls
call .venv\Scripts\activate.bat
ruff format src test
ruff check --fix src test
ty check
pyright

#python -m pytest test/ -v
echo.
echo All checks passed!