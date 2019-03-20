from args import parse_arguments
import multiprocessing
import os
import webbrowser
import time

params = parse_arguments()

if params.num_workers==-1:
    num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
else:
    num_workers = params.num_workers

launcher = 'tensorboard --logdir='

for worker in range(num_workers):
    launcher+='worker_'+str(worker)+':train_'+str(worker)+'/,'


if __name__ == '__main__':
    os.system(launcher)
    time.sleep(3)
    webbrowser.open('http://localhost:6006', new=2) 
