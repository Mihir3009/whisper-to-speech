# Shell script for make sampling frequency of the all wav file at 16kHz 

#/bin/sh
for entry in `ls /media/mihir/Dysarthia/Mirali/TORGO/New_F/* | sort -V `;do
fname=`basename $entry .wav`
echo $fname
sox -r 16000 -b 16 -c 1 $entry /media/mihir/Dysarthia/Mirali/TORGO/New_F/train/$fname.wav 
done
exit

