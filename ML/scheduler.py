from apscheduler.schedulers.blocking import BlockingScheduler
import datetime
import subprocess

def run_script():
    print(f"Running script at {datetime.datetime.now()}")
    subprocess.call(["python3", "run_all_model.py"])

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    # Schedule the job to run at 6:30 PM on weekdays (Monday through Friday)
    scheduler.add_job(run_script, 'cron', day_of_week='mon-fri', hour=18, minute=30)
    
    try:
        print("Scheduler started. Press Ctrl+C to exit.")
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")
