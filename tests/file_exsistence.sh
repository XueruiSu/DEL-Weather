FILE="/nfs/weather/era5_valid/1982/1/record_1.bin"  
if test -f "$FILE"; then  
    echo "$FILE exists."  
else  
    echo "$FILE does not exist."  
fi  
