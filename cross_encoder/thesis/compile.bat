@echo off
echo Starting LaTeX compilation...

echo Running first xelatex pass...
xelatex -synctex=1 -interaction=nonstopmode main.tex

echo Running biber...
biber main

echo Running second xelatex pass...
xelatex -synctex=1 -interaction=nonstopmode main.tex

echo Running final xelatex pass...
xelatex -synctex=1 -interaction=nonstopmode main.tex

echo Compilation complete!
pause