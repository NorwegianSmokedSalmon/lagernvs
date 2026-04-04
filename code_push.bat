@echo off
if "%~1"=="" (
    echo Error: Please provide a commit message.
    echo Usage: code_push.bat "your commit message"
    exit /b 1
)

echo Adding changes to git...
git add .

echo Committing with message: "%~1"
git commit -m "%~1"

echo Pushing to remote...
git push

echo Successfully pushed to repository!
