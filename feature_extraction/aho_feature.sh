
# shell script for extracting the features

#!/bin/sh
# ahocoder feature extraction Set low and high f0 range


speakers=([1]=M04 [2]=F03 [3]=F02 [4]=M09 [5]=F05 [6]=M08 [7]=M10 [8]=M12) 

arr=(1 2 3 4 5 6 7 8) 

  
# loops iterate through a set of values until the list (arr) is exhausted 
for i in "${arr[@]}"
do
	for entry in `ls /media/maitreya/Dysarthia/dysarthic_interspeech/UA/speaker_specific/data/${speakers[$i]}/$1/$2/*.wav`; do

	#echo $entry
	fname=`basename $entry .wav`
	echo $fname

	# convert and save wav file to 16k hz sampling frequency
	sox -r 16000 -b 16 -c 1 $entry /media/maitreya/Dysarthia/dysarthic_interspeech/scripts/tmp/$fname.wav 

	# feaure extraction : f0, mcc, fv 
	./ahocoder16_64 /media/maitreya/Dysarthia/dysarthic_interspeech/scripts/tmp/$fname.wav /media/maitreya/Dysarthia/dysarthic_interspeech/UA/speaker_specific/features/${speakers[$i]}/$1/$3/f0/$fname.f0 /media/maitreya/Dysarthia/dysarthic_interspeech/UA/speaker_specific/features/${speakers[$i]}/$1/$3/mcc/$fname.mcc /media/maitreya/Dysarthia/dysarthic_interspeech/UA/speaker_specific/features/${speakers[$i]}/$1/$3/fv/$fname.fv

	rm -r /media/maitreya/Dysarthia/dysarthic_interspeech/scripts/tmp/*.wav

	done
done
exit