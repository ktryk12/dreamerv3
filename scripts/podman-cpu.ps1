<#
.SYNOPSIS
  Podman wrapper for the CPU-first DreamerV3 image.

.DESCRIPTION
  Thin wrapper around `podman build` / `podman run` for Containerfile.podman.
  Logs are written to the named volume dreamerv3-logs so they survive the
  container.

.PARAMETER Command
  build  Build the image from Containerfile.podman.
  smoke  Short Crafter run with the debug config; verifies the agent trains.
  train  Full Crafter training run.
  shell  Interactive bash inside the image.
  logs   Print the contents of the log volume.
  clean  Remove the image and the log volume.

.EXAMPLE
  .\scripts\podman-cpu.ps1 build
  .\scripts\podman-cpu.ps1 smoke
#>
[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [ValidateSet('build', 'smoke', 'train', 'shell', 'logs', 'clean')]
  [string]$Command = 'build',

  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Rest
)

$ErrorActionPreference = 'Stop'

$Image = 'dreamerv3-cpu'
$Volume = 'dreamerv3-logs'
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Invoke-Podman {
  param([string[]]$Arguments)
  Write-Host "podman $($Arguments -join ' ')" -ForegroundColor DarkGray
  & podman @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "podman exited with code $LASTEXITCODE"
  }
}

function Assert-Machine {
  $running = & podman machine list --format '{{.Running}}' 2>$null
  if ($LASTEXITCODE -ne 0) {
    throw 'podman is not available on PATH.'
  }
  if ($running -notcontains 'true') {
    Write-Host 'Starting podman machine...' -ForegroundColor Yellow
    Invoke-Podman @('machine', 'start')
  }
}

Assert-Machine

switch ($Command) {
  'build' {
    Invoke-Podman @('build', '-f', (Join-Path $Root 'Containerfile.podman'), '-t', $Image, $Root)
  }
  'smoke' {
    # Tiny network, 4 envs, a few thousand steps: enough to prove that the
    # world model and the actor critic take gradient steps on CPU.
    Invoke-Podman @(
      'run', '--rm', '-v', "${Volume}:/logdir", $Image,
      'python', 'dreamerv3/main.py',
      '--logdir', '/logdir/smoke',
      '--configs', 'crafter', 'debug',
      '--run.steps', '2000',
      '--jax.platform', 'cpu'
    )
  }
  'train' {
    $extra = if ($Rest) { $Rest } else { @() }
    Invoke-Podman (@(
      'run', '--rm', '-it', '-v', "${Volume}:/logdir", $Image,
      'python', 'dreamerv3/main.py',
      '--logdir', '/logdir/train',
      '--configs', 'crafter',
      '--jax.platform', 'cpu'
    ) + $extra)
  }
  'shell' {
    Invoke-Podman @('run', '--rm', '-it', '-v', "${Volume}:/logdir", '--entrypoint', 'bash', $Image)
  }
  'logs' {
    Invoke-Podman @('run', '--rm', '-v', "${Volume}:/logdir", $Image, 'ls', '-R', '/logdir')
  }
  'clean' {
    & podman rmi $Image
    & podman volume rm $Volume
  }
}
