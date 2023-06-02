# ICS project repository

This is a rework of https://github.com/saimj7/People-Counting-in-Real-Time with some additions connected with illuminance estimation

## Usage 

First install requirements with `pip install -r requirements.txt`

Edit `mylib/config.py`:

- set `url` as in example to use IP-camera or left it `None` to use a laptop camera
- set `ShowVideo = True` to have real-time video with rectangles around people crossing the entrance
- set `Timer = True` and change `SchedulerStartTime, SchedulerTimeLimit` to make app run at a given time daily for given amount of seconds

Finally, run `python3 run.py`. Use `Ctrl+C` to exit. You can also run `python3 logger.py` to check written logs

## TODO:

- Make logs look more beautiful
- Try using cv2 BGR2HVS instead of manual color conversion
- Make reading logs more handy
- Refactoring + rename `mylib`
- Optional bounds for video sending to web page
- optional video stream
- add locks
- t.isoformat(' ', 'seconds')
