$types = @("_vh_clean_2.ply")
$scenes = Get-Content scannetv2_val.txt

foreach ($type in $types) {
    foreach ($scene in $scenes) {
        echo "" | python download-scannet.py -o [dir] --id $scene --type $type --skip_existing
    }
}