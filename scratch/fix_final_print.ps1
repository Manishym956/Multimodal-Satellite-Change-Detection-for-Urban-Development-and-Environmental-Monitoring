$txt = Get-Content src\training\train_scenario2_sync_multiseed.py -Raw
$txt = $txt.Replace('print(f"\nSaved all metrics and checkpoints cleanly to {output_dir}")', 'print("\nSaved all metrics and checkpoints cleanly to", str(output_dir).encode("ascii", "replace").decode())')
$txt | Set-Content src\training\train_scenario2_sync_multiseed.py -Encoding utf8
