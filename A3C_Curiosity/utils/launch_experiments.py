import os

#os.system("python main.py --scenario deadly_corridor --num_processes 12 --actions all --max_episodes 1400 --no_render")

#os.system("python main.py --scenario deadly_corridor --num_processes 12 --actions all --max_episodes 1400 --use_curiosity --no_render")

os.system("python main.py --scenario defend_the_center --num_processes 12 --actions all --max_episodes 1600 --no_render")

os.system("python main.py --scenario defend_the_center --num_processes 12 --actions all --max_episodes 1600 --use_curiosity --no_render")

#os.system("python main.py --scenario my_way_home --num_processes 12 --actions single --max_episodes 1600 --use_curiosity --no_render")

os.system("sudo shutdown -h now")