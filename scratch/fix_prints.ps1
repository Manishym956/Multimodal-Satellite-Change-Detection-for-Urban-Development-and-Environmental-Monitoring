$txt = Get-Content src\training\train_scenario2_sync_multiseed.py -Raw
$txt = $txt.Replace('print(f"Checkpoints:    {str(checkpoint_dir).encode(\"ascii\", \"replace\").decode()}")', 'print("Checkpoints:   ", str(checkpoint_dir).encode("ascii", "replace").decode())')
$txt = $txt.Replace('print(f"Output dir:     {str(output_dir).encode(\"ascii\", \"replace\").decode()}\n", flush=True)', 'print("Output dir:    ", str(output_dir).encode("ascii", "replace").decode(), "\n", flush=True)')
$txt | Set-Content src\training\train_scenario2_sync_multiseed.py -Encoding utf8
