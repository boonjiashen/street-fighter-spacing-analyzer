Runs computer vision algorithms on a Street Fighter 4 replay match

#Inspiration
Capcom's analysis of match between Snake Eyez (Zangief) and Fuudo (Fei Long)
[YouTube](http://youtu.be/dlIcud319Yk?t=15m42s)

#Run

I'm currently testing on a match in training stage since the background of
training stage is a lot easier. The match is [Infiltration (Gouken) vs Tokido
(Akuma)] (https://www.youtube.com/watch?v=XuqzhjKHhag).

1. Download [this video](https://www.youtube.com/watch?v=XuqzhjKHhag) as
   `data/my_video.mp4`, with your video downloader of choice
2. Run

<b></b>

    >> python3 train_SVM.py data/my_video.mp4 data/Infiltration_Gouken_vs_Tokido_Gouki.txt

#Results

[![replay](http://img.youtube.com/vi/37NeE0lTZc8/0.jpg)](https://www.youtube.com/watch?v=37NeE0lTZc8)
Date: Feb 18 2015
