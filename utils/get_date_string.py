from datetime import datetime

def get_date_string():
    now = datetime.now()
    return now.strftime("%m%d%Y-%H%M%S")
