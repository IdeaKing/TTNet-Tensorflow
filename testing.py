import time

def printProgressBar(
    iter, 
    total, 
    run_type,
    epoch = '', 
    ce = '', 
    wce = '', 
    dicebce = '',):
    """Training Progress Bar"""
    decimals = 1
    length = 20
    fill = '█'
    printEnd = "\r"

    percent = ("{0:." + str(decimals) + "f}").format(
            100 * (iter / float(total)))
    filledLength = int(length * iter // total)
    bar = fill * filledLength + '-' * (length - filledLength)

    print(f'\r {run_type} Epoch: {epoch} |{bar}| {percent}%' \
          f' {run_type} CE Loss: {ce} ' \
          f' {run_type} Weighted CE Loss: {wce} ' \
          f' {run_type} DICE-BCE Loss: {dicebce}', end = '\r')

    # Print New Line on Complete
    if iter == total: 
        # print()
        x = 1
            
step_size=30
for epoch in range(10):
    printProgressBar(
        iter=0, 
        total=step_size,
        run_type="Train")
    for iter in range(step_size):
        time.sleep(0.5)
        printProgressBar(
            iter=iter,
            total=step_size,
            run_type="Train",
            epoch=epoch+1,
            ce=str(iter),
            wce=str(iter),
            dicebce=str(iter))