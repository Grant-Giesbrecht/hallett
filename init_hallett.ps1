# Add this script to your powershell startup script to make the scripts and
# tools in this package accessible from any directory on your system.

# Get input arguments
param(
	[string]$ProjectPath,
	[bool]$Verbose
)

$ViewsparamScriptPath = Join-Path -Path $ProjectPath -ChildPath "tools\view_sparam.py"
function view_sparam {
	param(
		[Parameter(ValueFromRemainingArguments=$true)]
		[string[]]$args
	)
	
	python $ViewsparamScriptPath @args
}

# Success message if requested
if ($Verbose){
	Write-Output "Added hallett at path:" $ProjectPath
}