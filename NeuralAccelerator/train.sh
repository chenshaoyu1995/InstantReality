if [ $1 = "fixation" ]
then
    python3 train_fixation.py
fi

if [ $1 = "saccade" ]
then
    python3 train_saccade.py
fi

