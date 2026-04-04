param (
    [string]$CommitMessage
)

if ([string]::IsNullOrWhiteSpace($CommitMessage)) {
    Write-Host "Error: Please provide a commit message." -ForegroundColor Red
    Write-Host 'Usage: ./code_push.ps1 "your commit message"'
    exit 1
}

Write-Host "Adding changes to git..." -ForegroundColor Cyan
git add .

Write-Host "Committing with message: '$CommitMessage'" -ForegroundColor Cyan
git commit -m $CommitMessage

Write-Host "Pushing to remote..." -ForegroundColor Cyan
git push

Write-Host "Successfully pushed to repository!" -ForegroundColor Green
