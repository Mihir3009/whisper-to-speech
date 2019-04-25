
#!/bin/sh
# ahocoder feature extraction Set low and high f0 range


for entry in `ls /media/maitreya/Dysarthia/dysarthic_interspeech/UA/speaker_specific/result/$1/$2/converted_mcc/*.mcc `; do

fname=`basename $entry .mcc`

echo $fname

 
./ahodecoder16_64 /media/maitreya/Dysarthia/dysarthic_interspeech/UA/speaker_specific/features/$2/dysarthric/testing_feat/f0/$fname.f0 $entry /media/maitreya/Dysarthia/dysarthic_interspeech/UA/speaker_specific/result/$1/$2/converted_wav/$fname.wav

done
exit



