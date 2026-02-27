#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Starts both the FastAPI backend and React frontend development servers
.DESCRIPTION
    This script activates the Python virtual environment and starts the FastAPI backend,
    then starts the Vite frontend development server in parallel.
.EXAMPLE
    .\start-dev.ps1
#>

Write-Host "🚀 Starting NEPSE RL Dashboard Development Servers..." -ForegroundColor Green

# Get the script directory (project root)
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "📁 Project Root: $ProjectRoot" -ForegroundColor Cyan

# Start FastAPI backend in a new PowerShell window
Write-Host "🔧 Starting FastAPI backend server..." -ForegroundColor Yellow
$BackendJob = Start-Process powershell -ArgumentList @(
    "-NoExit", 
    "-Command", 
    "& { Set-Location '$ProjectRoot'; Write-Host '🐍 Backend server starting on http://localhost:8000' -ForegroundColor Green; uv run python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000 }"
) -PassThru

# Wait a moment for backend to start
Start-Sleep -Seconds 2

# Start Vite frontend in another new PowerShell window
Write-Host "⚡ Starting Vite frontend server..." -ForegroundColor Yellow
$FrontendJob = Start-Process powershell -ArgumentList @(
    "-NoExit", 
    "-Command", 
    "& { Set-Location '$ProjectRoot\frontend'; Write-Host '🌐 Frontend server starting on http://localhost:5173' -ForegroundColor Green; pnpm dev }"
) -PassThru

Write-Host "`n✅ Development servers started!" -ForegroundColor Green
Write-Host "🔗 Backend API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "🔗 Frontend UI: http://localhost:5173" -ForegroundColor Cyan
Write-Host "`n📋 Process IDs:" -ForegroundColor Gray
Write-Host "   Backend:  $($BackendJob.Id)" -ForegroundColor Gray
Write-Host "   Frontend: $($FrontendJob.Id)" -ForegroundColor Gray

Write-Host "`n🛑 To stop servers, close the PowerShell windows or press Ctrl+C in each" -ForegroundColor Yellow
Write-Host "🌟 Happy coding!" -ForegroundColor Magenta

# Optional: Open browser after a short delay
Start-Sleep -Seconds 3
Write-Host "🌐 Opening browser..." -ForegroundColor Gray
Start-Process "http://localhost:5173"