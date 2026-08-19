[CmdletBinding()]
param(
    [string]$ContainerName = "qwen38-b70-docker-test",
    [string]$ImageName = "qwen38-b70-docker:2026.08.18",
    [string]$ModelDirectory = (Join-Path $PSScriptRoot "models\Qwen3.8-27B-GPTQ-Int4"),
    [int]$ReadyTimeoutMinutes = 10
)

$ErrorActionPreference = "Stop"
$docker = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
if (-not (Test-Path -LiteralPath $docker)) {
    $docker = (Get-Command docker.exe -ErrorAction Stop).Source
}
. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-QwenDocker -DockerPath $docker

Write-Host "Windows container devised and tested by Ian Hudson - aitesthive.com"
Write-Host ""
Write-Host "[Docker] Starting the Qwen3.8-27B server on the Intel Arc Pro B70."
Write-Host "[Docker] Model: Qwen3.8-27B GPTQ INT4 with MTP4"
Write-Host "[Docker] Context: 100,000 tokens; explicit FP8 KV cache: 4.25 GiB"

if (-not (Test-Path -LiteralPath $ModelDirectory -PathType Container)) {
    throw "Model directory not found: $ModelDirectory"
}
$modelPath = (Resolve-Path -LiteralPath $ModelDirectory).Path

$containerNames = @(& $docker ps --all --filter "name=^/${ContainerName}$" --format '{{.Names}}')
$containerExists = $containerNames -contains $ContainerName
if ($containerExists) {
    $existing = (& $docker inspect --format '{{.State.Status}}' $ContainerName)
    if ($LASTEXITCODE -ne 0) { throw "Docker could not inspect the existing $ContainerName container." }
    if ($existing -eq "running") {
        Write-Host "[Docker] The existing container is already running; its compiled cache is being preserved."
    } else {
        Write-Host "[Docker] Restarting the existing container and preserving its compiled cache..."
        & $docker start $ContainerName | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Docker could not restart $ContainerName." }
    }
} else {
    Write-Host "[Docker] Creating the container. The first start can take around 10 minutes."
    Write-Host "[Docker] The model weights and XPU graphs need to load and compile, so you might want to go and make a cup of tea."
    $args = @(
        "run", "--detach", "--name", $ContainerName,
        "--device", "/dev/dxg", "--shm-size", "16g",
        "-p", "127.0.0.1:8000:8000",
        "-v", "/usr/lib/wsl/lib:/usr/lib/wsl/lib:ro",
        "-v", "/usr/lib/wsl/drivers:/usr/lib/wsl/drivers:ro",
        "--mount", "type=bind,source=$modelPath,target=/model,readonly",
        "-e", "LD_LIBRARY_PATH=/usr/lib/wsl/lib:/tmp/ucx_install/lib:/opt/venv/lib:/usr/local/lib",
        "-e", "VLLM_TARGET_DEVICE=xpu", "-e", "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE",
        "-e", "ZE_AFFINITY_MASK=0", "-e", "B70_MTP_BF16_DRAFT=1",
        "-e", "VLLM_XPU_ENABLE_XPU_GRAPH=1", "-e", "PYTORCH_ALLOC_CONF=expandable_segments:True",
        "-e", "CCL_ZE_IPC_EXCHANGE=sockets", "-e", "CCL_ATL_TRANSPORT=ofi",
        "-e", "CCL_ENABLE_SYCL_KERNELS=0", "-e", "CCL_TOPO_P2P_ACCESS=0",
        "-e", "CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0", "-e", "CCL_ZE_CACHE_OPEN_IPC_HANDLES=0",
        "-e", "SYCL_UR_USE_LEVEL_ZERO_V2=0", "-e", "TORCH_LLM_ALLREDUCE=1",
        "-e", "MTP_TOKENS=4", "-e", "MAX_MODEL_LEN=100000", "-e", "KV_CACHE_DTYPE=fp8",
        "-e", "MAX_NUM_SEQS=1", "-e", "GPU_MEMORY_UTILIZATION=0.75",
        "-e", "KV_CACHE_MEMORY_BYTES=4563402752", $ImageName
    )
    & $docker @args | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Docker container creation failed with exit code $LASTEXITCODE." }
}

$deadline = [DateTime]::UtcNow.AddMinutes($ReadyTimeoutMinutes)
$lastStage = ""
do {
    try {
        $models = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/models" -TimeoutSec 5
        if ($models.data.id -contains "qwen38") {
            Write-Host ""
            Write-Host "[Docker] Qwen3.8 is ready."
            Write-Host "[Docker] OpenAI-compatible endpoint: http://127.0.0.1:8000/v1"
            Write-Host "[Docker] Model name: qwen38"
            exit 0
        }
    } catch { }

    $status = (& $docker inspect --format '{{.State.Status}} {{.State.ExitCode}}' $ContainerName 2>$null)
    if ($status -notmatch '^running ') {
        & $docker logs --tail 80 $ContainerName
        throw "The Docker server exited before its API became ready ($status)."
    }

    $logs = (& $docker logs --tail 120 $ContainerName 2>&1) -join "`n"
    $stage = if ($logs -match 'Loading safetensors checkpoint shards:\s+(\d+)%') {
        "Loading model checkpoint shards ($($Matches[1])% of the current checkpoint pass)"
    } elseif ($logs -match 'torch\.compile') {
        "Compiling and warming XPU execution graphs"
    } elseif ($logs -match 'Initializing a V1 LLM engine') {
        "Initialising the vLLM engine"
    } else {
        "Preparing the container and checking the B70"
    }
    if ($stage -ne $lastStage) {
        Write-Host "[Docker] $stage"
        $lastStage = $stage
    }
    Write-Host ("[Docker] Still loading... {0:hh\:mm\:ss}" -f ([TimeSpan]::FromMinutes($ReadyTimeoutMinutes) - ($deadline - [DateTime]::UtcNow)))
    Start-Sleep -Seconds 60
} while ([DateTime]::UtcNow -lt $deadline)

throw "The Docker server did not become ready within $ReadyTimeoutMinutes minutes. It may still be loading; inspect it with: docker logs -f $ContainerName"
