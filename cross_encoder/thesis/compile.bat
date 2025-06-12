@echo off
echo Starting LaTeX compilation with latexmk...

latexmk -xelatex -synctex=1 -interaction=nonstopmode -file-line-error main.tex

echo Compilation complete!
pause