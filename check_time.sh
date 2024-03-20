date +"%T"
sudo docker run -v /home/ubuntu/:/wav/ varshitdusad/4p_feedback:0.0.1 python3.6 4P_feedback.py wav/audiospodcast_28218.wav
echo "$audiospodcast_28218.wav"
date +"%T"

date +"%T"
sudo docker run -v /home/ubuntu/:/wav/ varshitdusad/4p_feedback:0.0.1 python3.6 4P_feedback.py wav/audiospodcast_28420.wav
echo "$audiospodcast_28420.wav"
date +"%T"
