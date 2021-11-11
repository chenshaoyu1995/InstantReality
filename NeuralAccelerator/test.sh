if [ $1 = 'fixation' ]
then
    python3 test_fixation.py --triangle_id 506
fi

if [ $1 = 'saccade' ]
then
    python3 test_saccade.py --triangle_id 506
fi
